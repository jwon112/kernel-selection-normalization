import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "manuscript/mathematics/reproducibility/recalculate.py"
if not PATH.exists():
    PATH = ROOT / "recalculate.py"
spec = importlib.util.spec_from_file_location("math_recalculate", PATH)
R = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = R
spec.loader.exec_module(R)


def row(case, dice=.8, present=True, predicted=1):
    return dict(case_id=case, checkpoint="checkpoints/latest.pt", **{
        key + region: value for region in R.REGIONS for key, value in
        (("dice_", str(dice)), ("gt_positive_", str(present)), ("predicted_voxels_", str(predicted)))})


class RecalculationTests(unittest.TestCase):
    def test_allcase_empty_empty_and_false_positive(self):
        rr = [row("BraTS-GLI-00001-000", .8), row("BraTS-GLI-00002-000", present=False, predicted=0)]
        self.assertAlmostEqual(R.macro(rr), .8)
        self.assertAlmostEqual(R.macro(rr, allcase=True), .9)
        rr[1] = row("BraTS-GLI-00002-000", present=False, predicted=5)
        self.assertAlmostEqual(R.macro(rr, allcase=True), .4)

    def test_patient_bootstrap_preserves_study_weighting(self):
        rr = [row("BraTS-GLI-00001-000", .2), row("BraTS-GLI-00001-001", .4), row("BraTS-GLI-00002-000", .9)]
        point, draws = R.cluster_scores(rr, ["BraTS-GLI-00001", "BraTS-GLI-00002"], np.array([[0, 1], [0, 0], [1, 1]]))
        self.assertAlmostEqual(point, .5)
        np.testing.assert_allclose(draws, [.5, .3, .9], rtol=0, atol=1e-15)

    def test_duplicate_or_wrong_fold_rejected(self):
        one, two = row("BraTS-GLI-00001-000"), row("BraTS-GLI-00002-000")
        for rr in ([one, one], [one, two]):
            with self.assertRaises(ValueError):
                R.groups(rr, {one["case_id"]})

    def test_invalid_flag_nonfinite_and_nonterminal_rejected(self):
        one = row("BraTS-GLI-00001-000")
        for key, value in (("dice_wt", "nan"), ("gt_positive_wt", "unknown")):
            with self.assertRaises(ValueError):
                R.macro([{**one, key: value}])
        with self.assertRaises(ValueError):
            R.groups([{**one, "checkpoint": "checkpoints/best.pt"}], {one["case_id"]})

    def test_packed_numeric_values_are_lossless_and_not_objects(self):
        values = (1, 2, 127, 0, 1, *[1.2345678901234567] * 12)
        packed = np.array([values], dtype=R.CHANNEL_DTYPE)
        unpacked = np.frombuffer(packed.tobytes(), dtype=R.CHANNEL_DTYPE)
        self.assertFalse(R.CHANNEL_DTYPE.hasobject)
        self.assertEqual(unpacked["variance"][0], 1.2345678901234567)
        self.assertEqual(unpacked["channel"][0], 127)
        self.assertLess(R.CHANNEL_DTYPE.itemsize * R.CHUNK_ROWS, 8 * 1024 * 1024)

    def test_path_escape_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "link").symlink_to("/etc/passwd")
            for name in ("../elsewhere", "/etc/passwd", "a\\b", ".secret", "link"):
                with self.assertRaises(ValueError):
                    R.safe_path(root, name)

    def test_scientific_snapshot_allows_only_provenance_change(self):
        original = {"score": .9, "input_sha256": {"main.tex": "old"}}
        self.assertEqual(R.science_digest(original), R.science_digest({**original, "input_sha256": {"main.tex": "new"}}))
        self.assertNotEqual(R.science_digest(original), R.science_digest({**original, "score": .8}))

    def test_changed_numeric_file_rejected_even_if_summary_unchanged(self):
        evidence = {"score": .9, "input_sha256": {"original.csv": "source-pin"}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            content = b"case,dice\none,0.9\n"
            (root / "one.csv").write_bytes(content)
            manifest = dict(scientific_evidence_sha256=R.science_digest(evidence),
                channel_dtype=R.CHANNEL_DTYPE.descr, studies=["BraTS-GLI-00001-000"],
                original_sources={"original.csv": "source-pin"},
                files={"one.csv": dict(bytes=len(content), rows=1, sha256=hashlib.sha256(content).hexdigest())})
            (root / "manifest.json").write_text(json.dumps(manifest))
            R.Data(root, evidence)
            (root / "one.csv").write_bytes(b"case,dice\none,0.8\n")
            with self.assertRaises(ValueError):
                R.Data(root, evidence)

    def test_zero_positive_bootstrap_draw_is_not_silently_dropped(self):
        rr = [row("BraTS-GLI-00001-000", .8), row("BraTS-GLI-00002-000", present=False)]
        with self.assertRaises(ValueError):
            R.cluster_scores(rr, ["BraTS-GLI-00001", "BraTS-GLI-00002"], np.array([[1, 1]]))


if __name__ == "__main__":
    unittest.main()
