#!/usr/bin/env python3
"""Build an allowlisted Mathematics-only delivery archive, without uploading it."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[3]
PAPER = "manuscript/mathematics"
BLOCKED_SUFFIXES = {".pt", ".pth", ".ckpt", ".npy", ".npz", ".nii", ".gz", ".jsonl"}
MAX_FILE_BYTES = 8 * 1024 * 1024
CODE = (
    "src/train_run.py", "src/evaluate_selector_interventions.py",
    "src/evaluate_stage_selector_interventions.py", "src/evaluate_kernel_normalization.py",
    "src/kernel_normalization_diagnostics.py", "src/analyze_kernel_stage_replication.py",
    "src/analyze_kernel_removal_pairs.py", "src/prepare_brats_multidomain_v1.py",
    "src/prepare_brats_multidomain_v2_anatomical.py", "src/prepare_brats2023_gli_benchmark.py",
    "src/seg3d/__init__.py", "src/seg3d/data.py", "src/seg3d/losses.py",
    "src/seg3d/reporting.py", "src/seg3d/tracking.py", "src/seg3d/complexity.py",
    "src/seg3d/multikernel.py", "src/seg3d/models.py", "src/seg3d/branch_p.py",
    "src/seg3d/model_zoo/__init__.py", "src/seg3d/model_zoo/registry.py",
)
PROTOCOLS = (
    "configs/kernel_stage_fivefold_twoseed_v1/protocol.json",
    "configs/kernel_stage_fivefold_twoseed_v1/seed2026.json",
    "configs/kernel_stage_fivefold_twoseed_v1/seed20261001.json",
    "configs/kernel_stage_thirdseed_v1/protocol.json",
    "configs/kernel_stage_thirdseed_v1/seed20261011.json",
    "configs/kernel_normalization_audit_v1/protocol.json",
)
HELPERS = {
    "manuscript/tools/build_evidence_tables.py": {
        "ROOT", "REGIONS", "THREE", "FIVE", "sha", "finite", "present", "region_macro",
        "read_csv", "read_json", "config_differences", "brats_family",
    },
    "manuscript/tools/build_statistics.py": {"patient_id", "cluster_matrix", "fold_scores"},
}


def digest(content):
    return hashlib.sha256(content).hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or ".." in path.parts or "\\" in name
            or any(p.startswith(".") for p in path.parts)
            or path.suffix.lower() in BLOCKED_SUFFIXES
            or path.name.lower() in {"auth.json", "credentials", "openreview.token"}):
        raise ValueError("Disallowed archive path: " + name)
    return str(path)


def scoped_module(content, names):
    """Keep exact source spans of explicitly named helpers, not other papers' analyses."""
    text = content.decode()
    lines = text.splitlines(keepends=True)
    selected, found = [], set()
    for node in ast.parse(text).body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            selected.append("".join(lines[node.lineno - 1:node.end_lineno]))
            continue
        keys = set()
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            keys = {node.name}
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            keys = {t.id for t in targets if isinstance(t, ast.Name)}
        if keys & names:
            selected.append("".join(lines[node.lineno - 1:node.end_lineno]))
            found |= keys & names
    if found != names:
        raise ValueError("Missing scoped helper definitions: " + repr(names - found))
    return ("\n\n".join(selected) + "\n").encode()


def collect(root):
    root = root.resolve()
    payload, inventory = {}, {}
    evidence_path = root / PAPER / "generated/evidence.json"
    evidence = json.loads(evidence_path.read_text())
    recorded = evidence["input_sha256"]

    def add(source, target=None, transform=None):
        source, target = safe_name(source), safe_name(target or source)
        path = root / source
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError("Missing, symlinked or oversized package source: " + source)
        content = path.read_bytes()
        source_hash = digest(content)
        if source in recorded and source_hash != recorded[source]:
            raise ValueError("Source changed since manuscript evidence build: " + source)
        if transform:
            content = transform(content)
        if target in payload:
            raise ValueError("Duplicate archive target: " + target)
        payload[target] = content
        inventory[target] = dict(bytes=len(content), sha256=digest(content),
                                 source=source, source_sha256=source_hash,
                                 transformation="scoped AST extraction" if transform else "none")

    for name in CODE + PROTOCOLS:
        add(name)
    frozen_identity = {}
    for name in ("src/train_run.py", "src/seg3d/multikernel.py", "src/seg3d/models.py",
                 "src/seg3d/data.py", "src/seg3d/reporting.py",
                 "src/evaluate_stage_selector_interventions.py"):
        source = "results/kernel_stage_fivefold_twoseed_v1/code/" + name
        archived = (root / source).read_bytes()
        if digest(archived) != inventory[name]["source_sha256"]:
            raise ValueError("Primary study frozen-code identity mismatch: " + name)
        frozen_identity[name] = dict(frozen_source=source, sha256=digest(archived))
    for name in sorted(n for n in recorded if n.startswith("configs/") and n.endswith(".json")):
        if name not in payload:
            add(name)
    for name, selected in HELPERS.items():
        add(name, transform=lambda content, selected=selected: scoped_module(content, selected))
    for name in ("build.py", "completed_diagnostics.py", "check_stage_robustness.py"):
        add(PAPER + "/tools/" + name)
    for name in ("test_mathematics_reporting.py", "test_mathematics_stage_robustness.py",
                 "test_mathematics_normalization_frequency.py", "test_multikernel.py"):
        add("tests/" + name)
    add(PAPER + "/main.tex")
    add(PAPER + "/main.pdf")
    add(PAPER + "/references.bib")
    for name in ("2026-10-10-reference-review.md", "2026-10-10-reference-query.json",
                 "2026-10-10-reference-checks.json", "2026-10-11-review-response-query.json",
                 "2026-10-11-review-response.json", "2026-10-11-author-role-amendment.json",
                 "2026-10-11-shared-first-authorship.json", "2026-10-11-software-role-amendment.json"):
        add(PAPER + "/audits/" + name)
    # Exact extensions and a paper-only directory; no recursive repository scan.
    for path in sorted((root / PAPER / "generated").iterdir()):
        if path.suffix == ".tex" or path.name in {
            "evidence.json", "normalization_evidence.json", "run_reporting.json",
            "stage_robustness_check.json", "normalization.pdf", "interventions.pdf",
        }:
            add(path.relative_to(root).as_posix())
    for name in ("mdpi.cls", "mdpi.bst", "journalnames.tex", "logo-mdpi.eps",
                 "logo-mdpi-eps-converted-to.pdf", "logo-orcid.pdf", "logo-updates.eps"):
        add(PAPER + "/Definitions/" + name)
    add("environment/pip-jaewon-freeze.txt")
    for name in ("metadata.json", "splits.json"):
        add("data/processed/brats2023_gli_crop4/" + name, "data_contract/" + name)
    for study in ("kernel_stage_fivefold_twoseed_v1", "kernel_stage_thirdseed_v1",
                  "kernel_normalization_audit_v1", "mathematics_fold_completion_v1"):
        add("results/" + study + "/queue/plan.json", "protocols/" + study + "_executed_plan.json")
    add("results/kernel_normalization_audit_v1/cpu_preflight/sensitivity_selection.json",
        "data_contract/normalization_sensitivity_selection.json")
    add(PAPER + "/reproducibility/README.md", "README.md")
    add(PAPER + "/reproducibility/verify_package.py", "verify_package.py")
    add(PAPER + "/reproducibility/recalculate.py", "recalculate.py")
    add(PAPER + "/reproducibility/prepare_gli.py", "prepare_gli.py")
    add(PAPER + "/tools/export_analysis_data.py", "numerical_exporter_source.py")
    add("tests/test_mathematics_recalculate.py")
    numerical_base = PAPER + "/reproducibility/analysis_data"
    numerical = json.loads((root / numerical_base / "manifest.json").read_text())
    scientific = {k: v for k, v in evidence.items() if k != "input_sha256"}
    if digest(json.dumps(scientific, sort_keys=True, allow_nan=False).encode()) != numerical["scientific_evidence_sha256"]:
        raise ValueError("Numerical export belongs to a different scientific snapshot")
    # Explicitly named numeric files only; retain the general 8-MB limit and
    # checkpoint/image/JSONL bans. Packed .bin records cannot contain pickle.
    for name, record in numerical["files"].items():
        if PurePosixPath(name).suffix not in {".csv", ".bin"}:
            raise ValueError("Unexpected numerical export type")
        target = "analysis_data/" + safe_name(name)
        add(numerical_base + "/" + name, target)
        if inventory[target]["sha256"] != record["sha256"] or inventory[target]["bytes"] != record["bytes"]:
            raise ValueError("Changed numerical export")
    for source, expected in numerical["original_sources"].items():
        if recorded.get(source) != expected:
            raise ValueError("Numerical export source disagrees with manuscript ledger")
    add(numerical_base + "/manifest.json", "analysis_data/manifest.json")
    add(PAPER + "/tools/package_reproducibility.py", "package_exporter_source.py")
    # A minimal installation list, without unrelated editable/third-party packages.
    packages = {"numpy", "scipy", "torch", "matplotlib", "nibabel", "thop"}
    freeze = payload["environment/pip-jaewon-freeze.txt"].decode().splitlines()
    requirements = [line for line in freeze if line.split("==", 1)[0] in packages]
    if len(requirements) != len(packages):
        raise ValueError("Incomplete numerical/training environment pins")
    content = ("# Recorded study versions; select the PyTorch CUDA/CPU index for your host.\n"
               + "\n".join(requirements) + "\n").encode()
    payload["requirements-paper.txt"] = content
    inventory["requirements-paper.txt"] = dict(bytes=len(content), sha256=digest(content),
        source="environment/pip-jaewon-freeze.txt", source_sha256=recorded["environment/pip-jaewon-freeze.txt"],
        transformation="explicit six-package selection")
    # Track relocation explicitly rather than mistaking a relocated file for absence.
    by_source = {row["source"]: target for target, row in inventory.items()}
    for source, target in numerical["csv_sources"].items():
        # These CSVs are exact copies; channel JSONL remains excluded and has
        # a separate, explicitly transformed numerical representation.
        by_source[source] = "analysis_data/" + target
    missing = {k: v for k, v in recorded.items() if k not in by_source}
    manifest = dict(schema_version=2, scope="Mathematics paper-specific reproducibility package",
        distribution_status="versioned release artifact; exporter does not perform or verify deposition",
        release_url="https://github.com/jwon112/kernel-selection-normalization/releases/tag/v1.0.0",
        files=inventory, original_input_sha256=recorded, original_inputs_not_included=missing,
        original_source_to_package_path=by_source,
        primary_study_frozen_code_identity=frozen_identity,
        numerical_record_export=dict(manifest="analysis_data/manifest.json",
            files=len(numerical["files"]), original_sources=len(numerical["original_sources"]),
            csv_originals=len(numerical["csv_sources"]),
            scientific_evidence_sha256=numerical["scientific_evidence_sha256"]),
        boundaries=["No raw/processed images, checkpoints, channel JSONL, other manuscripts, credentials or Git metadata",
                    "SHA256 proves file identity against this manifest, not authenticity or scientific correctness",
                    "Numerical-record reanalysis recomputes bootstrap intervals, selector associations and channel quantiles; not independent image inference or retraining",
                    "Full original channel JSONL is replaced by lossless typed numeric measurements for the reported analyses",
                    "Dataset membership is included as de-identified BraTS challenge IDs; respect provider terms",
                    "Generic shared code contains optional model constructors; only paper-configured models are packaged/tested",
                    "Historical protocol status fields are retained as written, not treated as current execution status"])
    payload["PACKAGE_MANIFEST.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    return payload, manifest


def write_archive(payload, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create: a prior delivery package is never silently overwritten.
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, content in sorted(payload.items()):
            info = zipfile.ZipInfo(safe_name(name), date_time=(2026, 10, 10, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content)
    return digest(output.read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload, manifest = collect(args.root)
    checksum = write_archive(payload, args.out)
    sidecar = args.out.with_suffix(args.out.suffix + ".sha256")
    with sidecar.open("x") as stream:
        stream.write(checksum + "  " + args.out.name + "\n")
    print(json.dumps(dict(archive=str(args.out), files=len(payload), bytes=args.out.stat().st_size,
                         sha256=checksum, omitted_original_inputs=len(manifest["original_inputs_not_included"])), indent=2))


if __name__ == "__main__":
    main()
