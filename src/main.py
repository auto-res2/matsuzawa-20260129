import os
import subprocess
import sys

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.train",
        f"results_dir={cfg.results_dir}",
        f"mode={cfg.mode}",
        f"run={cfg.run}",
    ]
    env = os.environ.copy()
    env["HYDRA_FULL_ERROR"] = "1"
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
