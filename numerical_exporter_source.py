#!/usr/bin/env python3
"""Export immutable numerical records, never images/checkpoints, to a NEW folder."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "manuscript/mathematics/reproducibility"))
from recalculate import (BRANCHES, CHANNEL_DTYPE, CHUNK_ROWS, METRICS, MODES,
                         PRIMITIVES, science_digest)


def export(root, out):
    evidence = json.loads((root / "manuscript/mathematics/generated/evidence.json").read_text())
    ledger = evidence["input_sha256"]
    out.mkdir(parents=True, exist_ok=False)
    files, sources, csv_sources = {}, {}, {}
    splits = json.loads((root / "data/processed/brats2023_gli_crop4/splits.json").read_text())
    studies = sorted(c for split in splits for c in split["val"])
    if len(studies) != len(set(studies)) or len(studies) != 1251:
        raise ValueError("Unexpected study mapping")
    case_index = {case: i for i, case in enumerate(studies)}

    def record(path, rows, source):
        content = path.read_bytes()
        files[path.relative_to(out).as_posix()] = dict(bytes=len(content), rows=rows,
            sha256=hashlib.sha256(content).hexdigest(), source=source)
        if len(content) > 8 * 1024 * 1024:
            raise ValueError("Oversized numerical export")

    def accept(source, digest):
        if ledger.get(source) != digest:
            raise ValueError("Source changed since manuscript analysis: " + source)
        sources[source] = digest

    (out / "csv").mkdir()
    # Only CSV sources already used and pinned by the paper's evidence builder.
    for i, source in enumerate(sorted(s for s in ledger if s.endswith(".csv"))):
        path = root / source
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Unsafe original CSV source")
        content = path.read_bytes()
        accept(source, hashlib.sha256(content).hexdigest())
        target = out / "csv" / f"{i:04d}.csv"
        target.write_bytes(content)
        with target.open(newline="") as stream:
            count = sum(1 for _ in csv.DictReader(stream))
        record(target, count, source)
        csv_sources[source] = target.relative_to(out).as_posix()
    (out / "normalization").mkdir()
    jobs = []
    case_fields = ("case_id", "stage", "branch") + tuple("gt_positive_" + r for r in ("wt", "tc", "et")) + tuple(
        policy + "_dice_" + r for policy in ("drop", "renorm") for r in ("wt", "tc", "et")) + tuple("mask_disagreement_" + r for r in ("wt", "tc", "et"))
    for job in evidence["normalization"]["jobs"]:
        name = f"seed{job['base_seed']}_fold{job['fold']}"
        folder = out / "normalization" / name
        folder.mkdir()
        info = {key: job[key] for key in ("base_seed", "fold", "checkpoint_sha256", "sensitivity_studies")}
        info["modes"] = {}
        for mode in MODES:
            source = job["attempt"] + "/" + mode + "_cases.jsonl"
            digest = hashlib.sha256(); count = 0
            target = folder / (mode + "_cases.csv")
            with (root / source).open("rb") as stream, target.open("x", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=case_fields)
                writer.writeheader()
                for line in stream:
                    digest.update(line); row = json.loads(line)
                    writer.writerow({key: row[key] for key in case_fields}); count += 1
            accept(source, digest.hexdigest()); record(target, count, source)
            info["modes"][mode] = dict(cases=target.relative_to(out).as_posix(), channels={str(s): [] for s in (1, 2, 3)})
            source = job["attempt"] + "/" + mode + "_channels.jsonl"
            buffers = {stage: [] for stage in (1, 2, 3)}

            def flush(stage):
                buffer = buffers[stage]
                if not buffer:
                    return
                chunks = info["modes"][mode]["channels"][str(stage)]
                target = folder / f"{mode}_stage{stage}_{len(chunks):03d}.bin"
                values = np.array(buffer, dtype=CHANNEL_DTYPE)
                with target.open("xb") as stream:
                    stream.write(values.tobytes())
                record(target, len(values), source)
                chunks.append(target.relative_to(out).as_posix()); buffer.clear()

            digest = hashlib.sha256()
            with (root / source).open("rb") as stream:
                for line in stream:
                    digest.update(line); row = json.loads(line)
                    if row["zero_mass"] not in (True, False) or row["below_floor"] not in (True, False):
                        raise ValueError("Invalid numerical flag")
                    numbers = []
                    for key in METRICS + PRIMITIVES:
                        value = row[key]
                        if value is not None and not np.isfinite(value):
                            raise ValueError("Nonfinite source measurement")
                        numbers.append(np.nan if value is None else value)
                    stage = row["stage"]
                    if not 0 <= row["channel"] < {1: 128, 2: 64, 3: 32}[stage]:
                        raise ValueError("Invalid channel index")
                    buffers[stage].append((case_index[row["case_id"]], BRANCHES.index(row["branch"]), row["channel"],
                        int(row["zero_mass"]), int(row["below_floor"]), *numbers))
                    if len(buffers[stage]) == CHUNK_ROWS:
                        flush(stage)
            accept(source, digest.hexdigest())
            for stage in (1, 2, 3):
                flush(stage)
        jobs.append(info)
        print("Exported " + name, flush=True)
    manifest = dict(schema_version=1, scientific_evidence_sha256=science_digest(evidence),
        files=files, original_sources=sources, csv_sources=csv_sources, studies=studies,
        normalization=jobs, channel_dtype=CHANNEL_DTYPE.descr, branches=BRANCHES,
        null_policy="Original JSON null is represented by IEEE float64 NaN; finite values are lossless float64",
        channel_format="Packed little-endian records, no padding/header/pickle; read using recalculate.CHANNEL_DTYPE",
        boundary="Recorded numeric outputs only; BraTS challenge IDs preserve patient linkage. No images or checkpoints.")
    with (out / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True); stream.write("\n")
    return dict(directory=str(out), files=len(files)+1, bytes=sum(r["bytes"] for r in files.values()),
                original_sources=len(sources))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.root.resolve(), args.out.resolve()), indent=2))
