#!/usr/bin/env python3
"""Build a leakage-safe multi-domain BraTS dataset without duplicating 2024.

Domains:
  - brats2023_gli, brats2023_men, brats2023_ped: shared 3-class pre-treatment head
  - brats2024_ptg: 4-class post-treatment head, grouped by subject over timepoints
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np


TARGET = (192, 240, 192)
MODALITIES = ("t2f", "t1n", "t1c", "t2w")
ROOT = Path("/home/vlm/jaewon/Routable-Experts")
DEFAULT_OUT = ROOT / "data/processed/brats_multidomain_v1"
SOURCE_2024 = ROOT / "data/processed/brats2024_ptg"
ARCHIVES = {
    "brats2023_gli": [
        Path("/home/vision/datasets/BraTS2023/GLI/asnr-miccai-brats2023-gli-challenge-trainingdata.zip"),
    ],
    "brats2023_men": [
        Path("/home/vision/datasets/BraTS2023/MEN/asnr-miccai-brats2023-men-challenge-trainingdata.zip"),
        Path("/home/vision/datasets/BraTS2023/MEN/brats-men-train-fix-v4.zip"),
    ],
    "brats2023_ped": [
        Path("/home/vision/datasets/BraTS2023/PED/asnr-miccai-brats2023-ped-challenge-trainingdata.zip"),
    ],
}


def normalize(a: np.ndarray) -> np.ndarray:
    mask = a != 0
    out = np.zeros(a.shape, dtype=np.float32)
    values = a[mask]
    if values.size:
        out[mask] = np.clip((a[mask] - values.mean()) / max(float(values.std()), 1e-8), -5, 5)
    return out.astype(np.float16)


def center_crop_pad(a: np.ndarray) -> np.ndarray:
    out = np.zeros((a.shape[0], *TARGET), dtype=a.dtype)
    source_slices, target_slices = [], []
    for size, target in zip(a.shape[1:], TARGET):
        if size <= target:
            source_slices.append(slice(None))
            offset = (target - size) // 2
            target_slices.append(slice(offset, offset + size))
        else:
            offset = (size - target) // 2
            source_slices.append(slice(offset, offset + target))
            target_slices.append(slice(None))
    out[(slice(None), *target_slices)] = a[(slice(None), *source_slices)]
    return out


def load_nii_from_zip(zips: list[zipfile.ZipFile], members: dict[str, str], suffix: str) -> np.ndarray:
    """Later archives override earlier ones (used by the MEN correction archive)."""
    member = members[suffix]
    archive_index = int(member.split("::", 1)[0])
    archive_member = member.split("::", 1)[1]
    raw = gzip.decompress(zips[archive_index].read(archive_member))
    return np.asanyarray(nib.Nifti1Image.from_bytes(raw).dataobj)


def archive_cases(paths: list[Path]) -> tuple[list[zipfile.ZipFile], dict[str, dict[str, str]]]:
    zips = [zipfile.ZipFile(path) for path in paths]
    cases: dict[str, dict[str, str]] = defaultdict(dict)
    suffixes = tuple([*MODALITIES, "seg"])
    for archive_index, archive in enumerate(zips):
        for member in archive.namelist():
            if not member.endswith(".nii.gz"):
                continue
            filename = Path(member).name
            for suffix in suffixes:
                token = f"-{suffix}.nii.gz"
                if filename.endswith(token):
                    case_id = filename[: -len(token)]
                    cases[case_id][suffix] = f"{archive_index}::{member}"
                    break
    incomplete = {case_id: sorted(set(suffixes) - set(files)) for case_id, files in cases.items() if set(files) != set(suffixes)}
    if incomplete:
        preview = list(incomplete.items())[:3]
        raise RuntimeError(f"Incomplete archive cases: {preview}")
    return zips, dict(cases)


def patient_id(domain: str, case_id: str) -> str:
    if domain == "brats2024_ptg":
        parts = case_id.split("-")
        if len(parts) != 4 or parts[:2] != ["BraTS", "GLI"]:
            raise ValueError(f"Unexpected 2024 ID: {case_id}")
        return "-".join(parts[:3])
    # 2023 training data have one baseline study per challenge case.
    return case_id


def make_splits(records: list[dict], folds: int, seed: int) -> tuple[list[dict], dict]:
    """Stratify domains, while treating every 2024 longitudinal subject as a group."""
    by_domain: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_domain[record["domain"]][record["subject_id"]].append(record["case_id"])

    buckets = [[] for _ in range(folds)]
    per_fold_domain_counts = [Counter() for _ in range(folds)]
    rng = random.Random(seed)
    for domain, groups in sorted(by_domain.items()):
        grouped = list(groups.items())
        rng.shuffle(grouped)
        grouped.sort(key=lambda item: len(item[1]), reverse=True)
        study_counts = [0] * folds
        for _, studies in grouped:
            fold = min(range(folds), key=lambda i: (study_counts[i], len(buckets[i])))
            buckets[fold].extend(studies)
            study_counts[fold] += len(studies)
            per_fold_domain_counts[fold][domain] += len(studies)

    all_ids = {r["case_id"] for r in records}
    record_by_id = {r["case_id"]: r for r in records}
    splits, audit_folds = [], []
    for fold, validation in enumerate(buckets):
        validation = sorted(validation)
        train = sorted(all_ids - set(validation))
        train_subjects = {(record_by_id[x]["domain"], record_by_id[x]["subject_id"]) for x in train}
        val_subjects = {(record_by_id[x]["domain"], record_by_id[x]["subject_id"]) for x in validation}
        overlap = train_subjects & val_subjects
        if overlap:
            raise RuntimeError(f"Subject leakage in fold {fold}: {sorted(overlap)[:3]}")
        splits.append({"train": train, "val": validation})
        audit_folds.append({
            "fold": fold,
            "train_cases": len(train), "validation_cases": len(validation),
            "train_subjects": len(train_subjects), "validation_subjects": len(val_subjects),
            "subject_overlap": 0,
            "validation_cases_by_domain": dict(sorted(Counter(record_by_id[x]["domain"] for x in validation).items())),
        })
    return splits, {"scheme": "5-fold stratified-by-domain grouped-CV", "seed": seed, "folds": audit_folds}


def sampling_plan(records: list[dict], splits: list[dict]) -> dict:
    record_by_id = {r["case_id"]: r for r in records}
    domains = sorted({r["domain"] for r in records})
    plan = {"policy": "equal domain probability; 2024 is additionally uniform-by-subject then uniform-by-study", "domains": domains, "folds": {}}
    for fold, split in enumerate(splits):
        train = split["train"]
        by_domain: dict[str, list[str]] = defaultdict(list)
        by_subject: dict[tuple[str, str], list[str]] = defaultdict(list)
        for case_id in train:
            record = record_by_id[case_id]
            by_domain[record["domain"]].append(case_id)
            by_subject[(record["domain"], record["subject_id"])].append(case_id)
        # Unnormalized per-case sampling weights. Sum within each domain is 1.
        weights = {}
        for domain, case_ids in by_domain.items():
            subject_groups = [cases for (d, _), cases in by_subject.items() if d == domain]
            for cases in subject_groups:
                for case_id in cases:
                    weights[case_id] = 1.0 / (len(subject_groups) * len(cases))
        plan["folds"][str(fold)] = {
            "domain_cases": {d: len(by_domain[d]) for d in domains},
            "domain_subjects": {d: len([1 for (x, _) in by_subject if x == d]) for d in domains},
            "per_case_weights": weights,
        }
    return plan


def symlink(source: Path, destination: Path) -> None:
    destination.symlink_to(os.path.relpath(source, destination.parent))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    out = args.out
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {out}")

    (out / "images").mkdir(parents=True)
    (out / "labels_pretreatment_3class").mkdir()
    (out / "labels_posttreatment_4class").mkdir()
    records: list[dict] = []

    for domain, archive_paths in ARCHIVES.items():
        zips, cases = archive_cases(archive_paths)
        try:
            for index, case_id in enumerate(sorted(cases), 1):
                members = cases[case_id]
                arrays = [load_nii_from_zip(zips, members, modality).astype(np.float32) for modality in MODALITIES]
                label = load_nii_from_zip(zips, members, "seg").astype(np.uint8)
                if len({a.shape for a in arrays}) != 1 or arrays[0].shape != label.shape:
                    raise RuntimeError(f"Shape mismatch in {domain}/{case_id}")
                image = center_crop_pad(np.stack([normalize(a) for a in arrays]))
                label = center_crop_pad(label[None])[0]
                np.save(out / "images" / f"{case_id}.npy", image)
                np.save(out / "labels_pretreatment_3class" / f"{case_id}.npy", label)
                records.append({"case_id": case_id, "subject_id": patient_id(domain, case_id), "domain": domain,
                                "head": "pretreatment_3class", "image": f"images/{case_id}.npy",
                                "label": f"labels_pretreatment_3class/{case_id}.npy"})
                if index % 100 == 0:
                    print(f"{domain}: {index}/{len(cases)}", flush=True)
        finally:
            for archive in zips:
                archive.close()

    source_metadata = json.loads((SOURCE_2024 / "metadata.json").read_text())
    for source in sorted((SOURCE_2024 / "images").glob("*.npy")):
        case_id = source.stem
        label_source = SOURCE_2024 / "labels_raw" / source.name
        if not label_source.exists():
            raise RuntimeError(f"Missing 2024 label: {label_source}")
        symlink(source, out / "images" / source.name)
        symlink(label_source, out / "labels_posttreatment_4class" / source.name)
        records.append({"case_id": case_id, "subject_id": patient_id("brats2024_ptg", case_id), "domain": "brats2024_ptg",
                        "head": "posttreatment_4class", "image": f"images/{case_id}.npy",
                        "label": f"labels_posttreatment_4class/{case_id}.npy"})

    records.sort(key=lambda r: (r["domain"], r["case_id"]))
    splits, audit = make_splits(records, args.folds, args.seed)
    (out / "cases.json").write_text(json.dumps(records, indent=2) + "\n")
    (out / "splits_joint_grouped_stratified.json").write_text(json.dumps(splits, indent=2) + "\n")
    (out / "split_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (out / "sampling_plan.json").write_text(json.dumps(sampling_plan(records, splits), indent=2) + "\n")
    metadata = {
        "dataset": "BraTS multi-domain v1",
        "shape": list(TARGET), "modalities": list(MODALITIES),
        "normalization": "per-case per-modality nonzero z-score clipped [-5,5]",
        "heads": {
            "pretreatment_3class": {"domains": ["brats2023_gli", "brats2023_men", "brats2023_ped"], "labels": ["background", "NETC", "SNFH", "ET"]},
            "posttreatment_4class": {"domains": ["brats2024_ptg"], "labels": ["background", "NETC", "SNFH", "ET", "RC"]},
        },
        "case_counts": dict(sorted(Counter(r["domain"] for r in records).items())),
        "source_2024_metadata": source_metadata,
        "storage": "2023 arrays materialized; 2024 arrays symlinked from brats2024_ptg",
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"out": str(out), "case_counts": metadata["case_counts"], "audit": audit}, indent=2), flush=True)


if __name__ == "__main__":
    main()
