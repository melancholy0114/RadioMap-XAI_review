"""Regression tests for multi-backbone construction and compatibility."""

import unittest

import torch
import yaml

from explanation import GradCAM
from model import (
    RadioUNetC,
    Restormer,
    build_model,
    checkpoint_metadata,
    get_gradcam_target_layer,
    get_model_name,
    validate_checkpoint_model,
)


class ModelFactoryTests(unittest.TestCase):
    @staticmethod
    def _tiny_restormer_config(include_name=True):
        config = {
            "inp_channels": 2,
            "out_channels": 1,
            "dim": 8,
            "num_blocks": [1, 1, 1, 1],
            "num_refinement_blocks": 1,
            "heads": [1, 1, 2, 4],
            "ffn_expansion_factor": 2.0,
            "bias": False,
            "LayerNorm_type": "WithBias",
        }
        if include_name:
            config["name"] = "restormer"
        return config

    def test_legacy_config_defaults_to_restormer(self):
        model = build_model(self._tiny_restormer_config(include_name=False))
        self.assertIsInstance(model, Restormer)
        self.assertEqual(get_model_name(model), "restormer")

        output = model(torch.randn(1, 2, 32, 32))
        self.assertEqual(tuple(output.shape), (1, 1, 32, 32))

    def test_radiounet_c_forward_and_backward(self):
        model = build_model(
            {
                "name": "radiounet_c",
                "inp_channels": 2,
                "out_channels": 1,
            }
        )
        self.assertIsInstance(model, RadioUNetC)
        self.assertEqual(get_model_name(model), "radiounet_c")

        inputs = torch.randn(1, 2, 64, 64, requires_grad=True)
        output = model(inputs)
        self.assertEqual(tuple(output.shape), (1, 1, 64, 64))
        self.assertTrue(torch.isfinite(output).all())
        output.mean().backward()
        self.assertIsNotNone(model.conv_up000[0].weight.grad)

    def test_radiounet_c_rejects_incompatible_spatial_size(self):
        model = build_model({"name": "radiounet_c"})
        with self.assertRaisesRegex(ValueError, "divisible by 64"):
            model(torch.randn(1, 2, 96, 96))

    def test_backbone_settings_cannot_be_mixed(self):
        with self.assertRaisesRegex(ValueError, "Unsupported radiounet_c"):
            build_model({"name": "radiounet_c", "dim": 48})

    def test_checkpoint_metadata_blocks_cross_backbone_loading(self):
        restormer = build_model(self._tiny_restormer_config())
        radiounet = build_model({"name": "radiounet_c"})

        checkpoint = checkpoint_metadata(radiounet)
        validate_checkpoint_model(checkpoint, radiounet)
        with self.assertRaisesRegex(ValueError, "Checkpoint backbone"):
            validate_checkpoint_model(checkpoint, restormer)

        # Project checkpoints written before model metadata were all Restormer.
        validate_checkpoint_model({}, restormer)

    def test_gradcam_target_is_available_for_both_backbones(self):
        restormer = build_model(self._tiny_restormer_config())
        radiounet = build_model({"name": "radiounet_c"})
        self.assertIs(get_gradcam_target_layer(restormer), restormer.refinement[-1])
        self.assertIs(get_gradcam_target_layer(radiounet), radiounet.conv_up00)

    def test_radiounet_c_gradcam_backward(self):
        model = build_model({"name": "radiounet_c"})
        attribution = GradCAM(model, device="cpu").explain(
            torch.randn(1, 2, 64, 64)
        )
        self.assertEqual(tuple(attribution.shape), (1, 1, 64, 64))
        self.assertTrue(torch.isfinite(attribution).all())

    def test_checked_in_configs_keep_backbones_and_outputs_separate(self):
        with open("configs/config.yaml", "r") as handle:
            restormer_config = yaml.safe_load(handle)
        with open("configs/config_radiounet.yaml", "r") as handle:
            radiounet_config = yaml.safe_load(handle)

        self.assertEqual(restormer_config["model"]["name"], "restormer")
        self.assertEqual(radiounet_config["model"]["name"], "radiounet_c")
        self.assertNotEqual(
            restormer_config["output"]["checkpoint_dir"],
            radiounet_config["output"]["checkpoint_dir"],
        )


if __name__ == "__main__":
    unittest.main()
