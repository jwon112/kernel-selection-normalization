"""CPU-only fixtures for descriptive normalization endpoint frequencies."""
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / "manuscript/mathematics/tools/completed_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("normalization_diagnostics", PATH)
DIAGNOSTICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAGNOSTICS)


class NormalizationFrequencyTests(unittest.TestCase):
    def test_strict_absolute_thresholds_and_group_counts(self):
        cells = [dict(renorm_minus_drop=delta, base_seed=seed, fold=fold, branch=branch)
                 for delta, seed, fold, branch in
                 [(0.001, 1, 0, "k3"), (-0.01, 1, 1, "k5"), (0.06, 2, 0, "k3")]]
        rows = DIAGNOSTICS.exceedance_summary(cells)
        self.assertEqual([r["count"] for r in rows], [2, 1, 1])
        self.assertEqual(rows[0]["total"], 3)
        self.assertAlmostEqual(rows[0]["proportion"], 2 / 3)
        self.assertEqual(rows[0]["groups"]["base_seed"], {"1": 1, "2": 1})
        self.assertEqual(rows[2]["groups"]["branch"], {"k3": 1, "k5": 0})
        self.assertEqual(rows[0]["cells"][0]["renorm_minus_drop"], -0.01)

    def test_zero_counts_preserve_denominator(self):
        row = DIAGNOSTICS.exceedance_summary(
            [dict(renorm_minus_drop=0., base_seed=1, fold=0, branch="k3")])[0]
        self.assertEqual((row["count"], row["total"], row["proportion"]), (0, 1, 0.))

    def test_empty_cells_are_rejected(self):
        with self.assertRaises(ValueError):
            DIAGNOSTICS.exceedance_summary([])


if __name__ == "__main__":
    unittest.main()
