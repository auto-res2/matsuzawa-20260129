import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import optuna
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from src.model import AuditSimulator, CalibrationDataset, CalibrationModel
from src.preprocess import get_or_create_metrics


@dataclass
class TrialResult:
    metrics: Dict[str, float]
    audit_samples: List[Tuple[torch.Tensor, torch.Tensor]]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def adjust_cfg_for_mode(cfg: DictConfig) -> None:
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        cfg.num_seeds = 1
        cfg.budgets = cfg.budgets[:1]
        cfg.training.epochs = 1
        cfg.training.batch_size = min(int(cfg.training.batch_size), 2)
        cfg.dataset.max_items = min(int(cfg.dataset.max_items), int(cfg.trial.max_items))
        cfg.model.max_new_tokens = min(int(cfg.model.max_new_tokens), int(cfg.trial.max_new_tokens))
        cfg.trial.max_budget = min(int(cfg.trial.max_budget), int(cfg.budgets[0]))
        cfg.trial.max_log_steps = min(int(cfg.trial.max_log_steps), 2)
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"


def build_effective_cfg(cfg: DictConfig) -> DictConfig:
    # The runs config is already merged by Hydra's defaults system
    adjust_cfg_for_mode(cfg)
    return cfg


def save_run_artifacts(cfg: DictConfig, best_hyperparams: Dict[str, Any]) -> None:
    run_dir = Path(cfg.results_dir) / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "run_config.yaml"
    config_path.write_text(OmegaConf.to_yaml(cfg))
    params_path = run_dir / "best_hyperparams.json"
    params_path.write_text(json.dumps(best_hyperparams, indent=2))


def train_calibrator(
    cfg: DictConfig,
    audit_samples: List[Tuple[torch.Tensor, torch.Tensor]],
    global_step: int,
    log_wandb: bool,
) -> float:
    dataset = CalibrationDataset(audit_samples)
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True)
    model = CalibrationModel(input_dim=dataset.feature_dim)
    model.train()

    with torch.no_grad():
        dummy_out = model(torch.zeros(2, dataset.feature_dim))
        assert dummy_out.shape == (2, 1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()
    loss_value = 0.0

    for batch_idx, (features, labels) in enumerate(loader):
        if batch_idx == 0:
            assert features.shape[0] == labels.shape[0]
            assert features.shape[1] == dataset.feature_dim
        if cfg.mode == "trial" and batch_idx >= 2:
            break
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        labels = labels.view(-1, 1)
        loss = criterion(logits, labels)

        aux_grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=False)
        aux_grad_norm = torch.sqrt(sum(g.detach().pow(2).sum() for g in aux_grads))

        loss.backward()
        grad_norm = 0.0
        for param in model.parameters():
            assert param.grad is not None
            grad_norm += float(param.grad.detach().abs().sum())
        assert grad_norm > 0.0

        optimizer.step()
        loss_value = float(loss.detach())
        if log_wandb:
            wandb.log({"calibration_loss": loss_value, "calibration_grad_norm": float(aux_grad_norm)}, step=global_step)

    return loss_value


def run_simulation(
    cfg: DictConfig,
    simulator: AuditSimulator,
    hyperparams: Dict[str, Any],
    log_wandb: bool,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {"per_budget": {}, "all_metrics": []}
    global_step = 0

    for budget in cfg.budgets:
        effective_budget = int(budget)
        if cfg.mode == "trial":
            effective_budget = min(int(cfg.trial.max_budget), int(budget))
        metric_list: List[Dict[str, float]] = []
        audit_samples_all: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for seed in range(cfg.num_seeds):
            trial_metrics, audit_samples = simulator.run_trial(
                seed=seed,
                budget=effective_budget,
                tau=cfg.tau,
                delta=cfg.delta,
                method=cfg.method,
                hyperparams=hyperparams,
                max_steps=2 if cfg.mode == "trial" else None,
            )
            audit_samples_all.extend(audit_samples)
            metric_list.append(trial_metrics)
            if log_wandb:
                log_data = {k: v for k, v in trial_metrics.items()}
                log_data.update({"budget": effective_budget, "seed": seed, "phase": "per_seed"})
                wandb.log(log_data, step=global_step)
            global_step += 1
            if cfg.mode == "trial" and global_step >= cfg.trial.max_log_steps:
                break

        metrics_mean: Dict[str, float] = {}
        for key in metric_list[0].keys():
            if key in {"selected_strategy", "true_best_strategy"}:
                continue
            metrics_mean[key] = float(torch.tensor([m[key] for m in metric_list]).mean())
        results["per_budget"][str(effective_budget)] = metrics_mean
        results["all_metrics"].append(metrics_mean)
        if log_wandb:
            metrics_mean_log = {k: v for k, v in metrics_mean.items()}
            metrics_mean_log["budget"] = effective_budget
            metrics_mean_log["phase"] = "per_budget_mean"
            wandb.log(metrics_mean_log, step=global_step)

        if audit_samples_all:
            calibration_loss = train_calibrator(cfg, audit_samples_all, global_step, log_wandb)
            results["per_budget"][str(effective_budget)]["calibration_loss"] = calibration_loss
        global_step += 1

        if cfg.mode == "trial" and global_step >= cfg.trial.max_log_steps:
            break

    return results


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg = build_effective_cfg(cfg)
    set_seed(cfg.seed)

    metrics = get_or_create_metrics(cfg)
    assert metrics["X_L"].shape == metrics["X_U"].shape
    assert metrics["X_L"].shape[1] == metrics["group_ids"].shape[0]
    simulator = AuditSimulator(metrics, cfg)
    assert simulator.S > 0 and simulator.N > 0 and simulator.G > 0

    best_hyperparams = OmegaConf.to_container(cfg.audit_policy, resolve=True)
    best_hyperparams.update(OmegaConf.to_container(cfg.alpha_split, resolve=True))
    best_hyperparams.update(OmegaConf.to_container(cfg.eprocess, resolve=True))

    if cfg.optuna.n_trials > 0:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            params = dict(best_hyperparams)
            for space in cfg.optuna.search_spaces:
                name = space.param_name
                if space.distribution_type == "uniform":
                    params[name] = float(trial.suggest_float(name, space.low, space.high))
                elif space.distribution_type == "categorical":
                    params[name] = trial.suggest_categorical(name, space.choices)
                else:
                    raise ValueError(f"Unsupported distribution: {space.distribution_type}")
            trial_results = run_simulation(cfg, simulator, params, log_wandb=False)
            primary = [m[cfg.primary_metric] for m in trial_results["all_metrics"]]
            return float(torch.tensor(primary).mean())

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=int(cfg.optuna.n_trials))
        best_hyperparams.update(study.best_params)

    save_run_artifacts(cfg, best_hyperparams)

    if cfg.wandb.mode != "disabled":
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=cfg.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",
            mode=cfg.wandb.mode,
        )

    results = run_simulation(cfg, simulator, best_hyperparams, log_wandb=(cfg.wandb.mode != "disabled"))

    if cfg.wandb.mode != "disabled":
        last_budget_key = list(results["per_budget"].keys())[-1]
        final_metrics = results["per_budget"][str(last_budget_key)]
        for k, v in final_metrics.items():
            wandb.summary[f"final_{k}"] = v

        for metric_name in [
            "worst_group_unsafe_selection_rate",
            "feasible_top1_accuracy",
            "abstention_rate",
            "audit_allocation_disparity",
            "utility_regret_under_gate",
        ]:
            vals = [m[metric_name] for m in results["all_metrics"] if metric_name in m]
            if not vals:
                continue
            if metric_name in {"worst_group_unsafe_selection_rate", "abstention_rate", "audit_allocation_disparity", "utility_regret_under_gate"}:
                wandb.summary[f"best_{metric_name}"] = float(min(vals))
                if metric_name == cfg.primary_metric:
                    wandb.summary[metric_name] = float(min(vals))
            else:
                wandb.summary[f"best_{metric_name}"] = float(max(vals))
                if metric_name == cfg.primary_metric:
                    wandb.summary[metric_name] = float(max(vals))

        print(f"WandB URL: {wandb.run.url}")
        wandb.finish()


if __name__ == "__main__":
    main()
