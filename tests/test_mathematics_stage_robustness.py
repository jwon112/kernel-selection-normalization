"""CPU-only statistical fixtures for the retrospective stage check."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
from scipy.stats import rankdata, spearmanr

PATH = Path(__file__).resolve().parents[1] / "manuscript/mathematics/tools/check_stage_robustness.py"
SPEC = importlib.util.spec_from_file_location("stage_robustness", PATH)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class StageRobustnessTests(unittest.TestCase):
    def test_spearman_matches_scipy_including_ties(self):
        rows = [dict(weight=x, outcome=y, stage=i // 4 + 1)
                for i, (x, y) in enumerate(zip([1, 2, 2, 4, 3, 6, 8, 7], [3, 1, 4, 8, 5, 5, 6, 9]))]
        expected = spearmanr([r["weight"] for r in rows], [r["outcome"] for r in rows]).statistic
        self.assertAlmostEqual(CHECK.association(rows, "outcome"), expected)

    def test_stage_confounded_positive_becomes_negative(self):
        rows = [dict(stage=s, weight=10*s+b, outcome=10*s-b) for s in (1, 2, 3) for b in range(4)]
        self.assertGreater(CHECK.association(rows, "outcome"), 0)
        self.assertAlmostEqual(CHECK.association(rows, "outcome", "stage_adjusted_ranks"), -1)
        self.assertAlmostEqual(CHECK.association(rows, "outcome", "within_stage_ranks"), -1)

    def test_adjustment_matches_dummy_regression(self):
        rows = [dict(stage=s, weight=(s*b+3*b*b) % 17, outcome=(4*s*s+b*b) % 19)
                for s in (1, 2, 3) for b in range(4)]
        design = np.array([[r["stage"] == s for s in (1, 2, 3)] for r in rows], dtype=float)
        residuals = []
        for field in ("weight", "outcome"):
            rank = rankdata([r[field] for r in rows])
            residuals.append(rank - design @ np.linalg.lstsq(design, rank, rcond=None)[0])
        expected = np.corrcoef(residuals)[0, 1]
        self.assertAlmostEqual(CHECK.association(rows, "outcome", "stage_adjusted_ranks"), expected)

    def test_constant_after_adjustment_is_undefined(self):
        rows = [dict(stage=s, weight=s, outcome=s) for s in (1, 2, 3) for _ in range(4)]
        self.assertIsNone(CHECK.association(rows, "outcome", "stage_adjusted_ranks"))

    def test_absolute_forced_comparator_and_tie(self):
        rows = [dict(stage=1, weight=b, drop=b, keep=-b) for b in range(4)]
        self.assertEqual(CHECK.contrast(rows, "drop", "keep")["comparison"], "tie")

    def test_negative_removal_is_not_absolute_valued(self):
        rows = [dict(stage=1, weight=b, drop=-b, keep=b) for b in range(4)]
        result = CHECK.contrast(rows, "drop", "keep")
        self.assertEqual(result["comparison"], "weaker")
        self.assertAlmostEqual(result["removal_minus_absolute_forced"], -2)

    def test_missing_or_duplicate_cell_rejected(self):
        rows = [dict(base_seed=s, fold_index=f, stage=t, branch=b,
                     weight=i, drop_effect=i, keep_dice=-i, renorm_effect=i,
                     allcase_drop_effect=i, allcase_keep_dice=-i)
                for s in CHECK.SEEDS for f in range(5) for t in CHECK.STAGES
                for i, b in enumerate(CHECK.BRANCHES)]
        CHECK.analyze_cells(rows)
        with self.assertRaises(ValueError):
            CHECK.analyze_cells(rows[:-1])
        with self.assertRaises(ValueError):
            CHECK.analyze_cells(rows[:-1] + [rows[0]])


if __name__ == "__main__":
    unittest.main()
