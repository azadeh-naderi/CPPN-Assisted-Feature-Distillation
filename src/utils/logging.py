import csv
import json
import logging
import pickle
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def _git_commit_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


class RunLogger:
    """Structured per-run result logging: CSV per-epoch rows, one JSON summary,
    and arbitrary saved artifacts (checkpoints, genomes, patterns). Replaces
    ad-hoc print() calls with files a paper's result tables can be built from.
    """

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._epoch_csv_path = self.run_dir / "training_log.csv"
        self._epoch_fieldnames: list[str] | None = None
        self._log = get_logger(self.__class__.__name__)

    def log_config(self, cfg: dict[str, Any]) -> None:
        with open(self.run_dir / "config_snapshot.json", "w") as f:
            json.dump(cfg, f, indent=2, default=str)

    def log_epoch(self, epoch: int, **metrics: Any) -> None:
        row = {"epoch": epoch, **metrics}
        is_new = not self._epoch_csv_path.exists()
        if self._epoch_fieldnames is None:
            self._epoch_fieldnames = list(row.keys())
        with open(self._epoch_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._epoch_fieldnames)
            if is_new:
                writer.writeheader()
            writer.writerow(row)
        self._log.info("epoch %s: %s", epoch, metrics)

    def log_summary(self, **metrics: Any) -> None:
        summary = {
            **metrics,
            "git_commit_sha": _git_commit_sha(),
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "platform": platform.platform(),
        }
        with open(self.run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

    def save_artifact(self, name: str, obj: Any) -> Path:
        path = self.run_dir / name
        if name.endswith(".pt"):
            torch.save(obj, path)
        elif name.endswith(".pkl"):
            with open(path, "wb") as f:
                pickle.dump(obj, f)
        elif name.endswith(".json"):
            with open(path, "w") as f:
                json.dump(obj, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported artifact extension for {name!r}")
        return path
