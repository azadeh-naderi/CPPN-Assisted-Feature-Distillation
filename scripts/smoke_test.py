"""Thin wrapper: runs the full mode matrix on tiny settings to verify the
pipeline end-to-end before taking real settings to a GPU/cluster. See the
plan's Verification section for exact settings (configs/smoke/*.yaml,
configs/neat/cppn_neat_smoke.cfg)."""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_baseline_sweep.py"),
        "--config",
        "configs/smoke/fashionmnist_lenet_smoke.yaml",
        "--modes",
        "all",
        "--seeds",
        "0",
        "--smoke",
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
