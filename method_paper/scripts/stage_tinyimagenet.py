"""Prepares Tiny-ImageNet for torchvision's ImageFolder (run once, on a login
node -- compute nodes have no internet):

    wget -P data http://cs231n.stanford.edu/tiny-imagenet-200.zip
    unzip -q data/tiny-imagenet-200.zip -d data
    python method_paper/scripts/stage_tinyimagenet.py

The training images already sit in train/<wnid>/images/. The validation
images are all in val/images/ with labels in val/val_annotations.txt; this
script links them into val_by_class/<wnid>/ (hard links, so no extra disk
space; falls back to copying) and checks the counts: 200 classes, 500
training and 50 validation images each.
"""

import argparse
import os
import shutil
from collections import Counter
from pathlib import Path


def stage(root: Path) -> None:
    train, val = root / "train", root / "val"
    if not train.is_dir() or not (val / "val_annotations.txt").exists():
        raise SystemExit(f"{root} doesn't look like an unzipped tiny-imagenet-200 (see the module docstring)")
    out = root / "val_by_class"
    labels = {}
    for line in (val / "val_annotations.txt").read_text().splitlines():
        if line.strip():
            name, wnid = line.split("\t")[:2]
            labels[name] = wnid

    linked = 0
    for name, wnid in labels.items():
        src, dst = val / "images" / name, out / wnid / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            continue
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        linked += 1

    train_counts = {d.name: len(list((d / "images").glob("*.JPEG"))) for d in train.iterdir() if d.is_dir()}
    val_counts = Counter(labels.values())
    problems = []
    if len(train_counts) != 200 or set(train_counts.values()) != {500}:
        problems.append(f"train: {len(train_counts)} classes, per-class counts {sorted(set(train_counts.values()))}")
    if set(val_counts) != set(train_counts) or set(val_counts.values()) != {50}:
        problems.append(f"val: {len(val_counts)} classes, per-class counts {sorted(set(val_counts.values()))}")
    print(f"linked {linked} validation images into {out}")
    print("OK: 200 classes, 500 train / 50 val each" if not problems else "PROBLEMS: " + "; ".join(problems))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/tiny-imagenet-200")
    stage(Path(parser.parse_args().root))


if __name__ == "__main__":
    main()
