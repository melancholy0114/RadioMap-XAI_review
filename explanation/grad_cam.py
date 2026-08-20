"""
Grad-CAM explainer for Radio Map Prediction model.

Generates class activation maps by using gradients flowing into the
final convolutional layer of each encoder/decoder level.
"""

import torch
import torch.nn as nn
import numpy as np

from model import get_gradcam_target_layer


class GradCAM:
    def __init__(self, model, target_layer=None, device="cuda"):
        self.model = model
        self.device = device

        # Each backbone registers a comparable late, full-resolution feature
        # layer through the shared model helper.
        if target_layer is None:
            target_layer = get_gradcam_target_layer(model)

        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def explain(self, inputs, target_channel=0):
        """
        Compute Grad-CAM attribution map.

        Args:
            inputs: (B, C, H, W) input tensor
            target_channel: output channel to explain

        Returns:
            cam: (B, 1, H, W) attribution map (upsampled to input size)
        """
        self.model.eval()
        h, w = inputs.shape[-2:]

        # Forward pass (without torch.no_grad to enable hook capture)
        outputs = self.model(inputs)

        # Backward pass: gradient of target channel sum
        self.model.zero_grad()
        score = outputs[:, target_channel].sum()
        score.backward(retain_graph=True)

        # Get gradients and activations
        gradients = self.gradients  # (B, C, h', w')
        activations = self.activations  # (B, C, h', w')

        # Global average pooling of gradients -> weights
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)

        # Weighted combination of activations
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (B, 1, h', w')
        cam = torch.relu(cam)

        # Upsample to input size
        cam = torch.nn.functional.interpolate(
            cam, size=(h, w), mode="bilinear", align_corners=False
        )

        # Normalize
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam

    def explain_sample(self, inputs, target_channel=0):
        """
        Explain a single sample.

        Returns:
            cam: (H, W) numpy array
        """
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)

        # Forward pass must NOT use torch.no_grad() for hooks to work
        outputs = self.model(inputs)

        self.model.zero_grad()
        score = outputs[:, target_channel].sum()
        score.backward(retain_graph=True)

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        h, w = inputs.shape[-2:]
        cam = torch.nn.functional.interpolate(
            cam, size=(h, w), mode="bilinear", align_corners=False
        )

        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam[0, 0].cpu().numpy()
