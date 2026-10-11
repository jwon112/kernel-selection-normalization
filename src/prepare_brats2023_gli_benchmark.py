#!/usr/bin/env python3
"""Build a leakage-safe BraTS 2023 GLI benchmark without copying voxels."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
import random


ROOT = Path("/home/vlm/jaewon/Routable-Experts")
SOURCE = ROOT / "data/processed/brats_multidomain_v2"
OUT = ROOT / "data/processed/brats2023_gli_crop4"


def patient_id(case_id: str) -> str:
    parts = case_id.split("-")
    if len(parts) != 4 or parts[:2] != ["BraTS", "GLI"] or not parts[2].isdigit() or not parts[3].isdigit():
        raise ValueError(f"Unexpected BraTS GLI ID: {case_id}")
    return "-".join(parts[:3])


def make_splits(case_ids: list[str], folds: int, seed: int):
    groups: dict[str, list[str]] = defaultdict(list)
    for case_id in case_ids: groups[patient_id(case_id)].append(case_id)
    ordered = list(groups.items()); random.Random(seed).shuffle(ordered); ordered.sort(key=lambda item: len(item[1]), reverse=True)
    buckets = [[] for _ in range(folds)]
    for _, studies in ordered:
        bucket = min(range(folds), key=lambda index: (len(buckets[index]), index))
        buckets[bucket].extend(studies)
    splits, audit = [], []
    all_cases = set(case_ids)
    for fold, val in enumerate(buckets):
        val = sorted(val); train = sorted(all_cases - set(val))
        train_patients = {patient_id(x) for x in train}; val_patients = {patient_id(x) for x in val}
        overlap = train_patients & val_patients
        if overlap: raise RuntimeError(f"Patient leakage fold {fold}: {sorted(overlap)[:3]}")
        splits.append({"train": train, "val": val})
        audit.append({"fold": fold, "train_cases": len(train), "val_cases": len(val), "train_patients": len(train_patients), "val_patients": len(val_patients), "patient_overlap": 0})
    return splits, audit


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--source", type=Path, default=SOURCE); parser.add_argument("--out", type=Path, default=OUT); parser.add_argument("--seed", type=int, default=2026); args = parser.parse_args()
    if args.out.exists(): raise FileExistsError(f"Refusing to overwrite {args.out}")
    records = json.loads((args.source / "cases.json").read_text())
    case_ids = sorted(r["case_id"] for r in records if r["domain"] == "brats2023_gli")
    if len(case_ids) != 1251: raise RuntimeError(f"Expected 1251 GLI studies, found {len(case_ids)}")
    splits, audit = make_splits(case_ids, folds=5, seed=args.seed)
    args.out.mkdir(parents=True)
    os.symlink(os.path.relpath(args.source / "images", args.out), args.out / "images")
    os.symlink(os.path.relpath(args.source / "labels_pretreatment_3class", args.out), args.out / "labels")
    (args.out / "splits.json").write_text(json.dumps(splits, indent=2) + "\n")
    metadata = {
        "dataset": "BraTS 2023 GLI benchmark, anatomical crop margin 4",
        "source_dataset": str(args.source), "case_count": len(case_ids), "patient_count": len({patient_id(x) for x in case_ids}),
        "shape": [192, 224, 192], "modalities": ["t2f", "t1n", "t1c", "t2w"],
        "normalization": "per-case per-modality nonzero z-score clipped [-5,5] after anatomical crop",
        "target_encoding": "brats_regions_from_atomic_123",
        "regions": {"WT": "atomic labels 1|2|3", "TC": "atomic labels 1|3", "ET": "atomic label 3"},
        "split": {"scheme": "5-fold patient-grouped CV", "patient_key": "BraTS-GLI-{patient}; study suffix excluded", "seed": args.seed, "folds": audit},
        "storage": "images and raw labels are symlinked from brats_multidomain_v2; region targets are converted on load",
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata["split"], indent=2))


if __name__ == "__main__": main()
