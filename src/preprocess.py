import json
import os
import re
from typing import Dict, List, Tuple

import torch
from datasets import load_dataset

from src.model import LLMGenerator, Strategy, compute_metrics_for_strategies, get_strategies


IDENTITY_BUCKETS = {
    "race": ["black", "white", "asian", "latino", "hispanic", "african", "european"],
    "religion": ["muslim", "christian", "jew", "hindu", "buddhist", "sikh"],
    "gender": ["woman", "women", "man", "men", "girl", "boy"],
    "lgbtq": ["gay", "lesbian", "trans", "queer", "bisexual"],
}


def detect_identity_bucket(text: str) -> str:
    text = text.lower()
    for bucket, terms in IDENTITY_BUCKETS.items():
        if any(re.search(rf"\b{re.escape(term)}\b", text) for term in terms):
            return bucket
    return "none"


def load_realtoxicity(cfg) -> List[Dict[str, str]]:
    try:
        dataset = load_dataset("allenai/real-toxicity-prompts", split="train", cache_dir=".cache/")
    except Exception as exc:
        raise RuntimeError(f"Failed to load RealToxicityPrompts dataset: {exc}") from exc
    items: List[Dict[str, str]] = []
    max_items = int(cfg.dataset.max_items)
    subgroup_field = str(cfg.dataset.preprocessing.subgroup_field)
    for ex in dataset:
        if len(items) >= max_items:
            break
        prompt = ex.get("prompt")
        if isinstance(prompt, dict):
            prompt_text = prompt.get("text", "")
        else:
            prompt_text = str(prompt)
        continuation = ex.get("continuation")
        if isinstance(continuation, list) and continuation:
            reference = continuation[0].get("text", prompt_text)
        else:
            reference = str(continuation) if continuation is not None else prompt_text
        if subgroup_field == "identity_bucket":
            group = detect_identity_bucket(prompt_text)
        else:
            group = str(ex.get(subgroup_field, "none"))
            if group in {"", "None", "nan"}:
                group = "none"
        items.append({"prompt": prompt_text, "reference": reference, "group": group})
    return items


def load_bold(cfg) -> List[Dict[str, str]]:
    try:
        dataset = load_dataset("AlexaAI/bold", split="train", cache_dir=".cache/")
    except Exception as exc:
        raise RuntimeError(f"Failed to load BOLD dataset: {exc}") from exc
    items: List[Dict[str, str]] = []
    max_items = int(cfg.dataset.max_items)
    subgroup_field = str(cfg.dataset.preprocessing.subgroup_field)
    for ex in dataset:
        if len(items) >= max_items:
            break
        prompt = str(ex.get("prompt", ""))
        reference = str(ex.get("wikipedia", ""))
        if subgroup_field in {"domain", "category"}:
            group = str(ex.get(subgroup_field, "unknown"))
        else:
            group = str(ex.get("domain", "unknown"))
        if group in {"", "None", "nan"}:
            group = "unknown"
        items.append({"prompt": prompt, "reference": reference, "group": group})
    return items


def build_group_mapping(items: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    groups = sorted({item["group"] for item in items})
    group_to_id = {g: i for i, g in enumerate(groups)}
    for item in items:
        item["group_id"] = group_to_id[item["group"]]
    return items, group_to_id


def load_or_generate_outputs(
    cfg,
    items: List[Dict[str, str]],
    strategies: List[Strategy],
) -> Dict[str, List[str]]:
    os.makedirs(".cache/completions", exist_ok=True)
    cache_key = f"{cfg.dataset.name}_{cfg.model.name}_{cfg.dataset.max_items}_{cfg.model.max_new_tokens}".replace("/", "_")
    outputs: Dict[str, List[str]] = {}

    generator = LLMGenerator(cfg)

    for strat in strategies:
        out_path = os.path.join(".cache/completions", f"{cache_key}_{strat.name}.json")
        if os.path.exists(out_path):
            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    outputs[strat.name] = json.load(f)
                continue
            except Exception:
                outputs.pop(strat.name, None)

        prompts = [strat.apply(item["prompt"]) for item in items]
        completions = generator.generate_batch(prompts, batch_size=cfg.training.batch_size)
        outputs[strat.name] = completions
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(completions, f)
        except Exception as exc:
            raise RuntimeError(f"Failed to save completions cache {out_path}: {exc}") from exc

    return outputs


def get_or_create_metrics(cfg) -> Dict[str, torch.Tensor]:
    os.makedirs(".cache/metrics", exist_ok=True)
    cache_key = f"{cfg.dataset.name}_{cfg.model.name}_{cfg.dataset.max_items}_{cfg.model.max_new_tokens}".replace("/", "_")
    cache_path = os.path.join(".cache/metrics", f"{cache_key}_metrics.pt")
    if os.path.exists(cache_path):
        try:
            return torch.load(cache_path)
        except Exception:
            pass

    if cfg.dataset.name.lower() == "realtoxicityprompts":
        items = load_realtoxicity(cfg)
    elif cfg.dataset.name.lower() == "bold":
        items = load_bold(cfg)
    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset.name}")

    items, group_to_id = build_group_mapping(items)

    strategies = get_strategies(cfg)
    outputs = load_or_generate_outputs(cfg, items, strategies)
    metrics = compute_metrics_for_strategies(items, strategies, outputs)
    metrics["group_ids"] = torch.tensor([item["group_id"] for item in items], dtype=torch.long)
    metrics["group_names"] = list(group_to_id.keys())

    try:
        torch.save(metrics, cache_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to save metrics cache {cache_path}: {exc}") from exc
    return metrics
