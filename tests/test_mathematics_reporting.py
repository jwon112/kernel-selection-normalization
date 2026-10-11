"""No-training checks for the kernel paper's reviewer-requested reporting."""
import copy
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "mathematics_build", ROOT / "manuscript/mathematics/tools/build.py")
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


class MathematicsReportingTests(unittest.TestCase):
    def models(self):
        cases = {
            "BraTS-GLI-00001-000": {"gt_positive_wt": "True", "gt_positive_tc": "True", "gt_positive_et": "True"},
            "BraTS-GLI-00001-001": {"gt_positive_wt": "True", "gt_positive_tc": "False", "gt_positive_et": "False"},
            "BraTS-GLI-00002-000": {"gt_positive_wt": "True", "gt_positive_tc": "True", "gt_positive_et": "True"},
        }
        return {"a": cases, "b": copy.deepcopy(cases)}

    def test_study_and_patient_denominators_are_distinct(self):
        counts = build.paired_schedule_counts(self.models(), 0)
        self.assertEqual((counts["studies"], counts["patients"]), (3, 2))
        self.assertEqual(counts["positive_reference"]["wt"], {"studies": 3, "patients": 2})
        self.assertEqual(counts["positive_reference"]["et"], {"studies": 2, "patients": 2})

    def test_model_specific_missing_case_is_not_silently_dropped(self):
        models = self.models()
        models["b"].pop("BraTS-GLI-00001-001")
        with self.assertRaisesRegex(ValueError, "Unpaired"):
            build.paired_schedule_counts(models, 0)

    def test_model_specific_reference_eligibility_is_rejected(self):
        models = self.models()
        models["b"]["BraTS-GLI-00001-001"]["gt_positive_et"] = "True"
        with self.assertRaisesRegex(ValueError, "reference eligibility"):
            build.paired_schedule_counts(models, 0)

    def test_invalid_reference_flag_is_rejected(self):
        models = self.models()
        models["a"]["BraTS-GLI-00001-001"]["gt_positive_et"] = "unknown"
        with self.assertRaisesRegex(ValueError, "Invalid"):
            build.paired_schedule_counts(models, 0)

    def cells(self):
        return [dict(fold=0, stage=stage, branch=branch, weight=index / 10,
                     drop_effect=-index / 100, renorm_effect=index / 100,
                     keep_dice=0.5 + index / 10)
                for stage in (1, 2, 3) for index, branch in enumerate(build.BRANCHES)]

    def test_ranges_use_four_branch_summaries_not_patient_scores(self):
        ranges = build.stage_ranges(self.cells())
        self.assertEqual(len(ranges), 3)
        self.assertEqual(ranges[0]["drop_effect"], [-0.03, 0.0])
        self.assertEqual(ranges[0]["keep_dice"], [0.5, 0.8])

    def test_incomplete_stage_branch_bank_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "four named branches"):
            build.stage_ranges(self.cells()[1:])


if __name__ == "__main__":
    unittest.main()
