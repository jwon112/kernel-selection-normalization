#!/usr/bin/env python3
"""Primary stage/branch estimands for the complete five-fold/two-seed matrix."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from itertools import combinations
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

BRANCHES = ("k3", "k5", "dilated_k3", "anisotropic")
REGIONS = ("wt", "tc", "et")


def read(path, hashes, root):
    hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def metric(records, allcase=False):
    means = []
    for region in REGIONS:
        scores = []
        for row in records:
            flag = row[f"gt_positive_{region}"]
            if flag not in ("True", "False"):
                raise ValueError("Invalid GT status")
            if flag == "True":
                score = float(row[f"dice_{region}"])
            elif allcase:
                score = float(int(row[f"predicted_voxels_{region}"]) == 0)
            else:
                continue
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Invalid Dice")
            scores.append(score)
        if not scores:
            raise ValueError("No positive references")
        means.append(float(np.mean(scores)))
    return float(np.mean(means))


def rho(cells, endpoint):
    x = [r["weight"] for r in cells]
    y = [r[endpoint] for r in cells]
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(spearmanr(x, y).statistic)


def analyze(root, plan_path=None, out=None):
    out = out or root / "results/kernel_stage_fivefold_twoseed_v1"
    plan_path = plan_path or out / "queue/plan.json"
    plan = json.loads(plan_path.read_text())
    seeds = plan["protocol"]["base_seeds"]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("At least two distinct seeds are required")
    models = len(seeds) * 5
    splits = json.loads((root / "data/processed/brats2023_gli_crop4/splits.json").read_text())
    hashes, cells, runs = {}, [], []
    endpoints = ("drop_effect", "renormalized_drop_effect", "keep_dice",
                 "allcase_drop_effect", "allcase_keep_dice")
    for seed in plan["protocol"]["base_seeds"]:
        config = json.loads((root / plan["configs"][str(seed)]).read_text())
        for fold in range(5):
            queue = root / plan.get("state_queues", {}).get(str(seed), str((out / "queue").relative_to(root)))
            status = json.loads((queue / f"seed{seed}_fold{fold}.json").read_text())
            if status["phase"] != "complete":
                raise ValueError(f"Incomplete {models}-run matrix")
            base = root / config["results_root"] / config["dataset"] / config["trainer"] / f"fold_{fold}"
            expected = set(splits[fold]["val"])
            weights = defaultdict(dict)
            for row in read(base / "metrics/selector_validation_final_epoch.csv", hashes, root):
                if row["stage"] == "aggregate":
                    continue
                if row["checkpoint"] != "checkpoints/latest.pt":
                    raise ValueError("Nonterminal selector")
                key = (row["stage"], row["branch"])
                if row["case_id"] in weights[key]:
                    raise ValueError("Duplicate selector")
                weights[key][row["case_id"]] = float(row["weight"])
            conditions = {}
            for kind, folder in (("drop", "stage_selector_interventions_final_epoch"),
                                 ("renorm", "stage_selector_interventions_reroute_final_epoch"),
                                 ("keep", "stage_selector_keep_one_final_epoch")):
                grouped = defaultdict(list)
                for row in read(base / "metrics" / folder / "cases.csv", hashes, root):
                    if row["checkpoint"] != "checkpoints/latest.pt":
                        raise ValueError("Nonterminal intervention")
                    grouped[row["condition"]].append(row)
                if len(grouped) != 13 or any(len(v) != len(expected) or {r["case_id"] for r in v} != expected for v in grouped.values()):
                    raise ValueError("Unpaired intervention conditions")
                conditions[kind] = grouped
            learned = metric(conditions["drop"]["learned"])
            learned_all = metric(conditions["drop"]["learned"], True)
            if any(not np.isclose(metric(conditions[k]["learned"]), learned, atol=1e-10, rtol=0) for k in ("renorm", "keep")):
                raise ValueError("Different learned baselines")
            this = []
            for stage in (1, 2, 3):
                for branch in BRANCHES:
                    ww = weights[(f"decoder_{stage}", branch)]
                    if set(ww) != expected:
                        raise ValueError("Incomplete selector studies")
                    drop = conditions["drop"][f"drop_decoder_{stage}_{branch}"]
                    keep = conditions["keep"][f"keep_decoder_{stage}_{branch}"]
                    renorm = conditions["renorm"][f"drop_decoder_{stage}_{branch}_and_reroute"]
                    cell = {"base_seed": seed, "effective_seed": seed + fold,
                            "fold": fold, "stage": stage, "branch": branch,
                            "weight": float(np.mean(list(ww.values()))), "learned_dice": learned,
                            "drop_effect": learned - metric(drop),
                            "renormalized_drop_effect": learned - metric(renorm),
                            "keep_dice": metric(keep),
                            "allcase_drop_effect": learned_all - metric(drop, True),
                            "allcase_keep_dice": metric(keep, True)}
                    cells.append(cell)
                    this.append(cell)
            runs.append({"base_seed": seed, "fold": fold, "studies": len(expected),
                         "patients": len({c.rsplit("-", 1)[0] for c in expected}),
                         "dice": learned, "allcase_dice": learned_all,
                         "correlations": {e: rho(this, e) for e in endpoints},
                         "within_stage": [{"stage": s, **{e: rho([c for c in this if c["stage"] == s], e) for e in endpoints}} for s in (1, 2, 3)]})
    seed_summary = []
    for seed in plan["protocol"]["base_seeds"]:
        subset = [c for c in cells if c["base_seed"] == seed]
        rr = [r for r in runs if r["base_seed"] == seed]
        seed_summary.append({"base_seed": seed, "equal_fold_dice": float(np.mean([r["dice"] for r in rr])),
                             "equal_fold_allcase_dice": float(np.mean([r["allcase_dice"] for r in rr])),
                             "descriptive_pooled_correlations": {e: rho(subset, e) for e in endpoints}})
    seed_agreement = []
    for fold in range(5):
        for stage in (1, 2, 3):
            winners = {}
            for seed in plan["protocol"]["base_seeds"]:
                subset = sorted((c for c in cells if c["base_seed"] == seed and c["fold"] == fold and c["stage"] == stage), key=lambda c: BRANCHES.index(c["branch"]))
                winners[str(seed)] = {e: max(subset, key=lambda c: c[e])["branch"] for e in ("weight", "drop_effect", "keep_dice")}
            comparisons = [{"base_seeds": [a, b],
                            "agreement": {e: winners[str(a)][e] == winners[str(b)][e]
                                          for e in ("weight", "drop_effect", "keep_dice")}}
                           for a, b in combinations(seeds, 2)]
            entry = {"fold": fold, "stage": stage, "winners": winners}
            if len(seeds) == 2:
                entry["agreement"] = comparisons[0]["agreement"]
            else:
                entry["pairwise_agreement"] = comparisons
                entry["all_seeds_agree"] = {e: len({w[e] for w in winners.values()}) == 1
                                            for e in ("weight", "drop_effect", "keep_dice")}
            seed_agreement.append(entry)
    destination = out / "analysis"
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "stage_branch_cells.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    result = {"studies": plan["studies"], "patients": plan["patients"], "models": models,
              "primary_endpoint": plan["protocol"]["primary_endpoint"], "runs": runs,
              "seed_summary": seed_summary, "seed_stage_winner_agreement": seed_agreement,
              "tie_rule": "first maximum in fixed branch order", "source_hashes": hashes}
    (destination / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = [f"# Five-fold / {len(seeds)}-seed kernel-stage replication", "",
             f"All {models} complete intervention grids were verified; terminal checkpoints are validated by the queue completion gate.",
             f"The {len(seeds)} seeds evaluate the same {plan['studies']} studies / {plan['patients']} patients; they are not additional cohorts.", "",
             "| Base seed | Fold | Dice | Weight/drop rho | Weight/keep rho |", "|---:|---:|---:|---:|---:|"]
    for r in runs:
        c = r["correlations"]
        lines.append(f"| {r['base_seed']} | {r['fold']} | {r['dice']:.6f} | {c['drop_effect']} | {c['keep_dice']} |")
    lines.extend(["", "See summary.json for within-stage, all-case, pooled descriptive results and seed winner agreement."])
    (destination / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"models": models, "analysis": str(destination), "seed_summary": seed_summary}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    analyze(args.root.resolve(), args.plan, args.out)
