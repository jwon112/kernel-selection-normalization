#!/usr/bin/env python3
"""Retrospective CPU-only stage sensitivity; never edits manuscript evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parents[1]
SEEDS = (2026, 20261001, 20261011)
STAGES = (1, 2, 3)
BRANCHES = ("k3", "k5", "dilated_k3", "anisotropic")
TOL = 1e-12


def correlation(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) != len(y) or len(x) < 2 or not np.isfinite([x, y]).all():
        raise ValueError("Invalid correlation inputs")
    if np.ptp(x) <= TOL or np.ptp(y) <= TOL:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def association(rows, endpoint, method="spearman"):
    x = np.asarray([r["weight"] for r in rows])
    y = np.asarray([r[endpoint] for r in rows])
    stages = np.asarray([r["stage"] for r in rows])
    if method == "within_stage_ranks":
        xr, yr = np.empty(len(rows)), np.empty(len(rows))
        for stage in sorted(set(stages)):
            mask = stages == stage
            xr[mask], yr[mask] = rankdata(x[mask]), rankdata(y[mask])
        x, y = xr, yr
    else:
        x, y = rankdata(x), rankdata(y)
        if method == "stage_adjusted_ranks":
            # Equivalent to residualizing each bank's global ranks on stage
            # indicator variables, then correlating the two residual vectors.
            for stage in sorted(set(stages)):
                mask = stages == stage
                x[mask] -= np.mean(x[mask])
                y[mask] -= np.mean(y[mask])
        elif method != "spearman":
            raise ValueError("Unknown association method")
    return correlation(x, y)


def contrast(rows, removal, forced, method="spearman"):
    rd, rf = association(rows, removal, method), association(rows, forced, method)
    if rd is None or rf is None:
        comparison, margin = "undefined", None
    else:
        margin = rd - abs(rf)
        comparison = "stronger" if margin > TOL else "weaker" if margin < -TOL else "tie"
    return dict(removal_rho=rd, forced_path_rho=rf,
                removal_minus_absolute_forced=margin, comparison=comparison,
                cells=len(rows))


def summarize(rows):
    result = {name: sum(r["comparison"] == name for r in rows)
              for name in ("stronger", "weaker", "tie", "undefined")}
    result["banks"] = len(rows)
    result["positive_removal"] = sum(r["removal_rho"] is not None and r["removal_rho"] > TOL for r in rows)
    for key in ("removal_rho", "forced_path_rho", "removal_minus_absolute_forced"):
        values = [r[key] for r in rows if r[key] is not None]
        result[key] = dict(min=float(np.min(values)), median=float(np.median(values)),
                           max=float(np.max(values))) if values else None
    result["exceptions"] = [r for r in rows if r["comparison"] != "stronger"]
    return result


def analyze_cells(cells):
    expected = {(seed, fold, stage, branch) for seed in SEEDS for fold in range(5)
                for stage in STAGES for branch in BRANCHES}
    actual = [(r["base_seed"], r["fold_index"], r["stage"], r["branch"]) for r in cells]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Missing, duplicate or unexpected stage cells")
    variants = {
        "primary": ("drop_effect", "keep_dice"),
        "renormalized_removal": ("renorm_effect", "keep_dice"),
        "all_case_sensitivity": ("allcase_drop_effect", "allcase_keep_dice"),
    }
    modes = [("all_stages", None, None, "spearman")]
    modes += [(f"within_stage_{s}", s, None, "spearman") for s in STAGES]
    modes += [(f"leave_stage_{s}_out", None, s, "spearman") for s in STAGES]
    modes += [(s, None, None, s) for s in ("stage_adjusted_ranks", "within_stage_ranks")]
    result = {}
    for variant, (removal, forced) in variants.items():
        output = {}
        for name, only, omitted, method in modes:
            records = []
            for seed in SEEDS:
                for fold in range(5):
                    subset = [r for r in cells if r["base_seed"] == seed and
                              r["fold_index"] == fold and (only is None or r["stage"] == only)
                              and r["stage"] != omitted]
                    records.append(dict(base_seed=seed, fold=fold,
                                        **contrast(subset, removal, forced, method)))
            output[name] = dict(method=method, only_stage=only, omitted_stage=omitted,
                                rows=records, summary=summarize(records))
        result[variant] = output
    return result


def reconstruct():
    # Import only for real-data work: statistical tests need no torch/files.
    import build

    cells, folds, pooled, _ = build.replication_data()
    evidence_path = PAPER / "generated/evidence.json"
    evidence = json.loads(evidence_path.read_text())
    for key, actual in (("stage_cells", cells), ("stage_folds", folds),
                        ("seed_pooled_correlations", pooled)):
        if actual != evidence[key]:
            raise ValueError("Raw reconstruction differs from manuscript: " + key)
    if "stage_sensitivity" in evidence and analyze_cells(cells) != evidence["stage_sensitivity"]:
        raise ValueError("Stage sensitivity differs from manuscript evidence")
    sources = dict(build.HASHES)
    sources[str(evidence_path.relative_to(ROOT))] = build.sha(evidence_path)
    for path in (Path(__file__).resolve(), PAPER / "tools/build.py",
                 ROOT / "manuscript/tools/build_evidence_tables.py",
                 ROOT / "manuscript/tools/build_statistics.py"):
        sources[str(path.relative_to(ROOT))] = build.sha(path)
    for name, digest in build.HASHES.items():
        if name in evidence["input_sha256"] and evidence["input_sha256"][name] != digest:
            raise ValueError("Input hash differs from completed manuscript: " + name)
    return cells, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Rebuild and compare without writing")
    args = parser.parse_args()
    cells, sources = reconstruct()
    result = dict(schema_version=1, status="complete", analysis_status="retrospective_descriptive",
                  units=dict(banks=15, folds=5, base_seeds=list(SEEDS), stages=3,
                             branches_per_stage=4, stage_branch_cells=len(cells)),
                  methods={
                      "spearman": "Pearson correlation of average-tie ranks within each bank/subset",
                      "stage_adjusted_ranks": "Rank all 12 cells within a bank; subtract each stage's mean rank separately for weight and endpoint; correlate residuals",
                      "within_stage_ranks": "Rank the four branches separately in each bank-stage; correlate the concatenated three rank vectors",
                      "comparison": "Signed removal association minus absolute forced-path association; ties within 1e-12",
                  },
                  boundaries=[
                      "No new training, inference, data, seed or independent confirmation",
                      "Folds overlap in training; seeds reuse the same validation patients; counts are descriptive, not independent replications for a binomial test",
                      "Stage adjustment is a rank-association sensitivity, not a causal intervention on depth",
                      "Within-stage correlations have four branches; large coefficients do not establish patient-wise selection accuracy",
                      "No p-values, patient confidence intervals or noninferiority claims are computed from aggregate cells",
                      "This check does not test whether one branch alone drives the association",
                  ],
                  analysis=analyze_cells(cells), input_sha256=sources)
    target = PAPER / "generated/stage_robustness_check.json"
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.check:
        if not target.exists() or target.read_text() != encoded:
            raise ValueError("Saved stage robustness check differs from reconstruction")
        print("PASS: raw CSV reconstruction, checkpoint/source hashes and stage sensitivities match")
    else:
        target.write_text(encoded)
        print("Wrote", target.relative_to(ROOT), "sha256=" + hashlib.sha256(encoded.encode()).hexdigest())
    for mode, detail in result["analysis"]["primary"].items():
        s = detail["summary"]
        print(mode, f"positive={s['positive_removal']}/15",
              f"stronger={s['stronger']}/15", f"ties={s['tie']}",
              f"weaker={s['weaker']}", "removal=" + str(s["removal_rho"]))


if __name__ == "__main__":
    main()
