#!/usr/bin/env python3
"""Create the anatomical-crop version of the multi-domain BraTS dataset.

Unlike v1, this never center-crops a raw field of view. Each case is cropped
to the union nonzero anatomical mask with a fixed margin and then only padded.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np

import prepare_brats_multidomain_v1 as common


ROOT = Path("/home/vlm/jaewon/Routable-Experts")
OUT = ROOT / "data/processed/brats_multidomain_v2"
RAW_2024 = Path("/home/vision/datasets/BraTS2024/GLI")
# V2's fixed canvas is intentionally distinct from the historical V1 canvas.
# Do not inherit V1's 192 x 240 x 192 target: the verified V2 cache and all
# downstream GLI checkpoints use 192 x 224 x 192.
TARGET = (192, 224, 192)


def anatomical_crop(image: np.ndarray, label: np.ndarray, margin: int = 4):
    """Crop to union nonzero anatomy, never using the label to choose bounds."""
    mask = np.any(image != 0, axis=0)
    coordinates = np.where(mask)
    if not coordinates[0].size:
        raise RuntimeError("Empty anatomical mask")
    slices = tuple(
        slice(max(0, int(axis.min()) - margin), min(size, int(axis.max()) + 1 + margin))
        for axis, size in zip(coordinates, image.shape[1:])
    )
    return image[(slice(None), *slices)], label[slices], slices


def pad_only(image: np.ndarray, label: np.ndarray):
    shape = image.shape[1:]
    if any(size > target for size, target in zip(shape, TARGET)):
        raise RuntimeError(
            f"Anatomical crop {shape} exceeds fixed target {TARGET}; refusing to truncate anatomy"
        )
    before = tuple((target - size) // 2 for size, target in zip(shape, TARGET))
    after = tuple(target - size - left for size, target, left in zip(shape, TARGET, before))
    return (
        np.pad(image, [(0, 0), *zip(before, after)], mode="constant"),
        np.pad(label, list(zip(before, after)), mode="constant"),
    )


def write_case(out: Path, case_id: str, arrays: list[np.ndarray], label: np.ndarray, label_dir: str, transforms: dict):
    image = np.stack(arrays).astype(np.float32)
    if len({a.shape for a in arrays}) != 1 or image.shape[1:] != label.shape:
        raise RuntimeError(f"Shape mismatch in {case_id}")
    image, label, slices = anatomical_crop(image, label)
    cropped_shape = image.shape[1:]
    image = np.stack([common.normalize(a) for a in image]).astype(np.float16)
    image, label = pad_only(image, label)
    np.save(out / "images" / f"{case_id}.npy", image)
    np.save(out / label_dir / f"{case_id}.npy", label.astype(np.uint8))
    transforms[case_id] = {
        "anatomical_bbox": [[s.start, s.stop] for s in slices],
        "cropped_shape": list(cropped_shape),
        "padding_before": [(target - size) // 2 for size, target in zip(cropped_shape, TARGET)],
    }


def process_2023(out: Path, records: list[dict], transforms: dict):
    for domain, archive_paths in common.ARCHIVES.items():
        zips, cases = common.archive_cases(archive_paths)
        try:
            for index, case_id in enumerate(sorted(cases), 1):
                members = cases[case_id]
                arrays = [common.load_nii_from_zip(zips, members, modality).astype(np.float32) for modality in common.MODALITIES]
                label = common.load_nii_from_zip(zips, members, "seg").astype(np.uint8)
                write_case(out, case_id, arrays, label, "labels_pretreatment_3class", transforms)
                records.append({"case_id": case_id, "subject_id": common.patient_id(domain, case_id), "domain": domain,
                                "head": "pretreatment_3class", "image": f"images/{case_id}.npy",
                                "label": f"labels_pretreatment_3class/{case_id}.npy"})
                if index % 100 == 0:
                    print(f"{domain}: {index}/{len(cases)}", flush=True)
        finally:
            for archive in zips:
                archive.close()


def process_2024(out: Path, records: list[dict], transforms: dict):
    cases = []
    for subset in ("training_data1_v2", "training_data_additional"):
        for case_dir in sorted((RAW_2024 / subset).iterdir()):
            if case_dir.is_dir() and (case_dir / f"{case_dir.name}-seg.nii.gz").exists():
                cases.append(case_dir)
    if len(cases) != 1621:
        raise RuntimeError(f"Expected 1621 BraTS2024 training studies, found {len(cases)}")
    for index, case_dir in enumerate(cases, 1):
        case_id = case_dir.name
        arrays = [np.asanyarray(nib.load(case_dir / f"{case_id}-{m}.nii.gz").dataobj, dtype=np.float32) for m in common.MODALITIES]
        label = np.asanyarray(nib.load(case_dir / f"{case_id}-seg.nii.gz").dataobj, dtype=np.uint8)
        write_case(out, case_id, arrays, label, "labels_posttreatment_4class", transforms)
        records.append({"case_id": case_id, "subject_id": common.patient_id("brats2024_ptg", case_id), "domain": "brats2024_ptg",
                        "head": "posttreatment_4class", "image": f"images/{case_id}.npy",
                        "label": f"labels_posttreatment_4class/{case_id}.npy"})
        if index % 100 == 0:
            print(f"brats2024_ptg: {index}/{len(cases)}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--margin", type=int, default=4)
    args = parser.parse_args()
    if args.margin != 4:
        raise ValueError("v2 is intentionally fixed at the agreed 4-voxel anatomical margin")
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.out}")
    for directory in ("images", "labels_pretreatment_3class", "labels_posttreatment_4class"):
        (args.out / directory).mkdir(parents=True, exist_ok=True)

    records, transforms = [], {}
    process_2023(args.out, records, transforms)
    process_2024(args.out, records, transforms)
    records.sort(key=lambda r: (r["domain"], r["case_id"]))
    splits, audit = common.make_splits(records, args.folds, args.seed)
    metadata = {
        "dataset": "BraTS multi-domain v2 anatomical crop",
        "shape": list(TARGET), "modalities": list(common.MODALITIES),
        "normalization": "per-case per-modality nonzero z-score clipped [-5,5] after anatomical crop",
        "crop": {"mask": "union nonzero across all four modalities", "margin_voxels": 4,
                 "policy": "anatomical crop then pad only; never truncate a crop"},
        "heads": {"pretreatment_3class": {"domains": ["brats2023_gli", "brats2023_men", "brats2023_ped"], "labels": ["background", "NETC", "SNFH", "ET"]},
                  "posttreatment_4class": {"domains": ["brats2024_ptg"], "labels": ["background", "NETC", "SNFH", "ET", "RC"]}},
        "case_counts": dict(sorted(Counter(r["domain"] for r in records).items())),
    }
    (args.out / "cases.json").write_text(json.dumps(records, indent=2) + "\n")
    (args.out / "spatial_transforms.json").write_text(json.dumps(transforms, indent=2) + "\n")
    (args.out / "splits_joint_grouped_stratified.json").write_text(json.dumps(splits, indent=2) + "\n")
    (args.out / "split_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (args.out / "sampling_plan.json").write_text(json.dumps(common.sampling_plan(records, splits), indent=2) + "\n")
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "case_counts": metadata["case_counts"], "audit": audit}, indent=2), flush=True)


if __name__ == "__main__":
    main()
