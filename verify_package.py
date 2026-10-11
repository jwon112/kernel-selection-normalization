#!/usr/bin/env python3
"""Verify the extracted delivery package; optionally recompute CPU summary analyses."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys


def verify_hashes(root):
    manifest = json.loads((root / "PACKAGE_MANIFEST.json").read_text())
    files = manifest["files"]
    for name, row in files.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError("Unsafe manifest path")
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Symlink/escaped package file: " + name)
        content = path.read_bytes()
        if len(content) != row["bytes"] or hashlib.sha256(content).hexdigest() != row["sha256"]:
            raise ValueError("Package file differs: " + name)
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    allowed_extra = {"PACKAGE_MANIFEST.json"}
    actual = {n for n in actual if "__pycache__" not in PurePosixPath(n).parts}
    if actual != set(files) | allowed_extra:
        raise ValueError("Missing/unlisted package files: " + repr(actual ^ (set(files) | allowed_extra)))
    return len(files)


def assert_close(actual, expected, label):
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError("Different keys: " + label)
        for key in expected:
            assert_close(actual[key], expected[key], label + "/" + key)
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError("Different lengths: " + label)
        for i, value in enumerate(expected):
            assert_close(actual[i], value, label + "/" + str(i))
    elif isinstance(expected, float):
        if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("Different numerical value: " + label)
    elif actual != expected:
        raise ValueError("Different value: " + label)


def scientific_check(root, output=None):
    paper = root / "manuscript/mathematics"
    sys.path[:0] = [str(paper / "tools"), str(root / "src")]
    import build
    import numpy as np
    from completed_diagnostics import exceedance_summary
    from check_stage_robustness import analyze_cells

    evidence = json.loads((paper / "generated/evidence.json").read_text())
    numeric, curves = build.numerical_probes()
    assert_close(numeric, evidence["numerical"], "numerical CPU probes")
    sensitivity = analyze_cells(evidence["stage_cells"])
    assert_close(sensitivity, evidence["stage_sensitivity"], "stage sensitivity")
    separate = json.loads((paper / "generated/stage_robustness_check.json").read_text())
    assert_close(sensitivity, separate["analysis"], "standalone sensitivity")
    norms = evidence["normalization"]
    if len(norms["cells"]) != 540:
        raise ValueError("Incomplete normalization summary grid")
    keys = {(c["mode"], c["base_seed"], c["fold"], c["stage"], c["branch"]) for c in norms["cells"]}
    expected = {(m, s, f, t, b) for m in ("primary_bf16", "sensitivity_bf16", "sensitivity_fp32")
                for s in build.STAGE_SEEDS for f in range(5) for t in (1, 2, 3) for b in build.BRANCHES}
    if keys != expected:
        raise ValueError("Duplicate/missing normalization summary cell")
    for row in norms["endpoints"]:
        selected = [c for c in norms["cells"] if (c["mode"], c["stage"]) == (row["mode"], row["stage"])]
        assert_close(exceedance_summary(selected), row["exceedances"], "exceedances")
        changes = [abs(c["renorm_minus_drop"]) for c in selected]
        assert_close(float(np.median(changes)), row["median_absolute_difference"], "median difference")
        assert_close(max(changes), row["maximum_absolute_difference"], "maximum difference")
    splits = json.loads((root / "data_contract/splits.json").read_text())
    seen, patients = set(), set()
    for split in splits:
        train, val = set(split["train"]), set(split["val"])
        train_patients = {c.rsplit("-", 1)[0] for c in train}
        val_patients = {c.rsplit("-", 1)[0] for c in val}
        if train & val or train_patients & val_patients or seen & val:
            raise ValueError("Study/patient overlap in supplied splits")
        if len(train | val) != 1251:
            raise ValueError("Unexpected cohort size")
        seen |= val
        patients |= val_patients
    if (len(splits), len(seen), len(patients)) != (5, 1251, 1133):
        raise ValueError("Incomplete fivefold study/patient coverage")
    selection = json.loads((root / "data_contract/normalization_sensitivity_selection.json").read_text())
    if selection["split_sha256"] != hashlib.sha256((root / "data_contract/splits.json").read_bytes()).hexdigest():
        raise ValueError("Normalization selection has a different split source")
    for fold, split in enumerate(splits):
        ids = split["val"]
        cohort = {c.rsplit("-", 1)[0] for c in ids}
        chosen = sorted(cohort, key=lambda p: hashlib.sha256(("kernel_normalization_audit_v1:" + p).encode()).hexdigest())[:2]
        studies = [c for c in ids if c.rsplit("-", 1)[0] in chosen]
        row = selection["selection"][fold]
        if row["patients"] != chosen or row["studies"] != studies:
            raise ValueError("Sensitivity selection differs from outcome-independent rule")
    generated = build.outputs(evidence, curves)
    for name, content in generated.items():
        if (paper / "generated" / name).read_text() != content:
            raise ValueError("Summary-to-table mismatch: " + name)
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
        destination = output / "generated"
        destination.mkdir()
        for name, content in generated.items():
            (destination / name).write_text(content)
        build.PAPER = output
        build.figures(evidence, curves)
    if build.torch.cuda.is_initialized():
        raise ValueError("Verification unexpectedly initialized CUDA")
    return dict(cpu_probes="passed", stage_cells=len(evidence["stage_cells"]),
                normalization_summary_cells=len(norms["cells"]), table_and_json_files=len(generated),
                split_studies=len(seen), split_patients=len(patients), cuda_initialized=False,
                boundary="Frozen-summary verification; not a raw-image, bootstrap-draw or channel-record reconstruction")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--scientific", action="store_true")
    parser.add_argument("--recalculate", action="store_true", help="Recompute study bootstrap, selector analyses and channel distributions from supplied numerical records")
    parser.add_argument("--rebuild-to", type=Path, help="Write tables/figures into a NEW directory")
    parser.add_argument("--report-to", type=Path, help="Save verification JSON to a NEW file outside the extracted package")
    args = parser.parse_args()
    root = args.root.resolve()
    result = dict(file_hashes_verified=verify_hashes(root))
    if args.scientific or args.rebuild_to:
        result.update(scientific_check(root, args.rebuild_to))
    if args.recalculate:
        from recalculate import recalculate
        result["numerical_record_reanalysis"] = recalculate(root)
    if args.report_to:
        if args.report_to.resolve().is_relative_to(root):
            raise ValueError("Keep verification output outside the immutable package")
        with args.report_to.open("x") as stream:
            json.dump(result, stream, indent=2); stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
