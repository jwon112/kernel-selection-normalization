import unittest

import torch

from seg3d.models import CanonicalUNet3D, CanonicalUNet3DMultiKernel
from seg3d.multikernel import BRANCH_SPECS, MultiKernelSelector3D


class MultiKernelSelectorTests(unittest.TestCase):
    def test_all_modes_preserve_shape_and_normalize_weights(self):
        x = torch.randn(4, 8, 8, 8, 8)
        settings = (
            ("fixed", {"fixed_branch": "k3"}),
            ("uniform", {}),
            ("sk", {}),
            ("soft", {}),
            ("moe", {"top_k": 2}),
        )
        for mode, kwargs in settings:
            with self.subTest(mode=mode):
                module = MultiKernelSelector3D(
                    channels=8,
                    mode=mode,
                    router_jitter=0.0,
                    **kwargs,
                )
                output = module(x)
                snapshot = module.selection_snapshot()
                self.assertEqual(output.shape, x.shape)
                self.assertIsNotNone(snapshot)
                torch.testing.assert_close(
                    snapshot["weights"].sum(dim=1),
                    torch.ones(x.shape[0]),
                )

    def test_sparse_top2_executes_only_selected_experts(self):
        module = MultiKernelSelector3D(
            channels=8,
            mode="moe",
            top_k=2,
            router_jitter=0.0,
        )
        with torch.no_grad():
            module.router[-1].weight.zero_()
            module.router[-1].bias.copy_(torch.tensor((4.0, 3.0, -4.0, -5.0)))

        x = torch.randn(4, 8, 8, 8, 8, requires_grad=True)
        output = module(x)
        (output.square().mean() + 0.01 * module.auxiliary_loss()).backward()

        snapshot = module.selection_snapshot()
        selected = snapshot["selected_mask"]
        self.assertTrue(selected[:, :2].all())
        self.assertFalse(selected[:, 2:].any())
        for index, name in enumerate(BRANCH_SPECS):
            gradient = module.branches[name].spatial.weight.grad
            if index < 2:
                self.assertIsNotNone(gradient)
                self.assertGreater(gradient.abs().sum().item(), 0)
            else:
                self.assertIsNone(gradient)

    def test_balanced_top2_auxiliary_loss_is_one(self):
        module = MultiKernelSelector3D(
            channels=4,
            mode="moe",
            top_k=2,
            router_jitter=0.0,
        )
        with torch.no_grad():
            module.router[-1].weight.zero_()
            module.router[-1].bias.zero_()

        module(torch.randn(4, 4, 4, 4, 4))
        self.assertAlmostEqual(module.auxiliary_loss().item(), 1.0, places=6)

    def test_dense_drop_intervention_supports_both_normalization_policies(self):
        module = MultiKernelSelector3D(channels=4, mode="uniform")
        x = torch.randn(2, 4, 4, 4, 4)

        with module.intervention("drop_branch", branch="k5", renormalize=False):
            module(x)
            raw = module.selection_snapshot()["weights"]
        self.assertTrue((raw[:, 1] == 0).all())
        torch.testing.assert_close(raw.sum(dim=1), torch.full((2,), 0.75))

        with module.intervention("drop_branch", branch="k5", renormalize=True):
            module(x)
            normalized = module.selection_snapshot()["weights"]
        self.assertTrue((normalized[:, 1] == 0).all())
        torch.testing.assert_close(normalized.sum(dim=1), torch.ones(2))
        self.assertIsNone(module._intervention)

    def test_sparse_drop_distinguishes_removal_from_rerouting(self):
        module = MultiKernelSelector3D(
            channels=4,
            mode="moe",
            top_k=2,
            router_jitter=0.0,
        )
        with torch.no_grad():
            module.router[-1].weight.zero_()
            module.router[-1].bias.copy_(torch.tensor((4.0, 3.0, 2.0, 1.0)))
        x = torch.randn(2, 4, 4, 4, 4)

        with module.intervention("drop_branch", branch="k3", renormalize=False):
            module(x)
            removed = module.selection_snapshot()["weights"]
        self.assertTrue((removed[:, 0] == 0).all())
        self.assertTrue((removed.sum(dim=1) < 1).all())
        self.assertEqual((removed > 0).sum(dim=1).tolist(), [1, 1])

        with module.intervention("drop_branch", branch="k3", renormalize=True):
            module(x)
            rerouted = module.selection_snapshot()["weights"]
        self.assertTrue((rerouted[:, 0] == 0).all())
        torch.testing.assert_close(rerouted.sum(dim=1), torch.ones(2))
        self.assertEqual((rerouted > 0).sum(dim=1).tolist(), [2, 2])

    def test_force_interventions_override_selector(self):
        module = MultiKernelSelector3D(
            channels=4,
            mode="moe",
            top_k=2,
            router_jitter=0.0,
        )
        x = torch.randn(2, 4, 4, 4, 4)
        with module.intervention("force_branch", branch="anisotropic"):
            module(x)
            forced = module.selection_snapshot()["weights"]
        self.assertEqual((forced > 0).sum(dim=1).tolist(), [1, 1])
        self.assertTrue((forced[:, 3] == 1).all())

        with module.intervention("force_uniform"):
            module(x)
            uniform = module.selection_snapshot()["weights"]
        torch.testing.assert_close(uniform, torch.full_like(uniform, 0.25))


class CanonicalUNetMultiKernelTests(unittest.TestCase):
    def test_fixed_k3_can_exactly_match_original_last_decoder_conv(self):
        features = (4, 8, 16, 32, 64)
        torch.manual_seed(7)
        baseline = CanonicalUNet3D(features=features)
        torch.manual_seed(7)
        fixed = CanonicalUNet3DMultiKernel(
            features=features,
            selection_mode="fixed",
            fixed_branch="k3",
        )

        baseline_state = baseline.state_dict()
        fixed_state = fixed.state_dict()
        for key, value in baseline_state.items():
            if key in fixed_state and fixed_state[key].shape == value.shape:
                fixed_state[key] = value
        fixed_state[
            "decoders.3.3.branches.k3.spatial.weight"
        ] = baseline_state["decoders.3.3.weight"]
        fixed.load_state_dict(fixed_state)

        baseline.eval()
        fixed.eval()
        x = torch.randn(1, 4, 32, 32, 32)
        with torch.no_grad():
            expected = baseline(x)
            actual = fixed(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_experiment_spec_records_comparison_contract(self):
        model = CanonicalUNet3DMultiKernel(
            features=(4, 8, 16, 32, 64),
            selection_mode="moe",
            top_k=2,
        )
        spec = model.experiment_spec()
        self.assertEqual(spec["backbone"], "CanonicalUNet3D")
        self.assertEqual(spec["mode"], "moe")
        self.assertEqual(spec["top_k"], 2)
        self.assertEqual(spec["selection_resolution"], "case")
        self.assertEqual(spec["branch_names"], list(BRANCH_SPECS))


if __name__ == "__main__":
    unittest.main()
