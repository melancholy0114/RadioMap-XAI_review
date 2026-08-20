"""CNN backbone for the RadioUNet_C (clean-city, no-measurement) setting.

This module intentionally contains only the RadioUNet implementation.  The
existing Restormer backbone remains in :mod:`model.radio_map_model` so the two
architectures can be selected independently through configuration.

The channel widths, kernel sizes, pooling schedule, skip connections, and
input reinjection follow the first U-Net in the authors' public RadioUNet
implementation.  RadioUNet_C uses two inputs: the complete city/building map
and the transmitter-location map.  The optional retrospective second U-Net
from the original WNet curriculum is not part of this backbone comparison.

References:
    Paper: https://arxiv.org/abs/1911.09002 (Figure/Table 2)
    Code: https://github.com/RonLevie/RadioUNet/blob/master/lib/modules.py
"""

import torch
import torch.nn as nn


def _conv_relu_pool(in_channels, out_channels, kernel_size, padding, pool_size):
    """Build the Conv-ReLU-Pool unit used by the original RadioUNet."""
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
        ),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(pool_size, stride=pool_size),
    )


def _upconv_relu(in_channels, out_channels, kernel_size, padding):
    """Build the stride-two transposed-convolution unit used by RadioUNet."""
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
        ),
        nn.ReLU(inplace=True),
    )


class RadioUNetC(nn.Module):
    """RadioUNet_C predictor for complete city maps without measurements.

    Inputs are ``(building_map, transmitter_heatmap)`` and the output is one
    normalized radio-map channel.  Six factor-two pooling operations require
    both spatial dimensions to be divisible by 64; RadioMapSeer's 256 x 256
    samples satisfy this requirement.
    """

    required_spatial_multiple = 64

    def __init__(self, inp_channels=2, out_channels=1):
        super().__init__()
        if inp_channels != 2:
            raise ValueError(
                "RadioUNet_C requires exactly 2 input channels: "
                "building map and transmitter heatmap"
            )
        if out_channels != 1:
            raise ValueError("RadioUNet_C requires exactly 1 output channel")

        self.inp_channels = inp_channels
        self.out_channels = out_channels

        # Contracting path. Pool size 1 denotes an extra convolution at the
        # same spatial resolution, matching the public RadioUNet topology.
        self.layer00 = _conv_relu_pool(inp_channels, 6, 3, 1, 1)
        self.layer0 = _conv_relu_pool(6, 40, 5, 2, 2)
        self.layer1 = _conv_relu_pool(40, 50, 5, 2, 2)
        self.layer10 = _conv_relu_pool(50, 60, 5, 2, 1)
        self.layer2 = _conv_relu_pool(60, 100, 5, 2, 2)
        self.layer20 = _conv_relu_pool(100, 100, 3, 1, 1)
        self.layer3 = _conv_relu_pool(100, 150, 5, 2, 2)
        self.layer4 = _conv_relu_pool(150, 300, 5, 2, 2)
        self.layer5 = _conv_relu_pool(300, 500, 5, 2, 2)

        # Expanding path. Encoder features are concatenated before each unit.
        self.conv_up5 = _upconv_relu(500, 300, 4, 1)
        self.conv_up4 = _upconv_relu(300 + 300, 150, 4, 1)
        self.conv_up3 = _upconv_relu(150 + 150, 100, 4, 1)
        self.conv_up20 = _conv_relu_pool(100 + 100, 100, 3, 1, 1)
        self.conv_up2 = _upconv_relu(100 + 100, 60, 6, 2)
        self.conv_up10 = _conv_relu_pool(60 + 60, 50, 5, 2, 1)
        self.conv_up1 = _upconv_relu(50 + 50, 40, 6, 2)
        self.conv_up0 = _upconv_relu(40 + 40, 20, 6, 2)

        # RadioUNet reinjects both raw input channels at full resolution.
        self.conv_up00 = _conv_relu_pool(
            20 + 6 + inp_channels,
            20,
            5,
            2,
            1,
        )
        self.conv_up000 = _conv_relu_pool(
            20 + inp_channels,
            out_channels,
            5,
            2,
            1,
        )

    @property
    def gradcam_target_layer(self):
        """Last full-resolution feature block used for Grad-CAM.

        Hook the block output after its in-place ReLU. Hooking the convolution
        directly would make PyTorch's full backward hook conflict with that
        in-place operation.
        """
        return self.conv_up00

    def _validate_input(self, input_tensor):
        if input_tensor.ndim != 4:
            raise ValueError(
                "RadioUNet_C input must have shape (batch, channels, height, width)"
            )
        if input_tensor.shape[1] != self.inp_channels:
            raise ValueError(
                f"RadioUNet_C expected {self.inp_channels} channels, "
                f"received {input_tensor.shape[1]}"
            )
        height, width = input_tensor.shape[-2:]
        multiple = self.required_spatial_multiple
        if height % multiple != 0 or width % multiple != 0:
            raise ValueError(
                "RadioUNet_C spatial dimensions must be divisible by "
                f"{multiple}; received {height}x{width}"
            )

    def forward(self, input_tensor):
        self._validate_input(input_tensor)

        layer00 = self.layer00(input_tensor)
        layer0 = self.layer0(layer00)
        layer1 = self.layer1(layer0)
        layer10 = self.layer10(layer1)
        layer2 = self.layer2(layer10)
        layer20 = self.layer20(layer2)
        layer3 = self.layer3(layer20)
        layer4 = self.layer4(layer3)
        layer5 = self.layer5(layer4)

        layer4_up = self.conv_up5(layer5)
        layer3_up = self.conv_up4(torch.cat([layer4_up, layer4], dim=1))
        layer20_up = self.conv_up3(torch.cat([layer3_up, layer3], dim=1))
        layer2_up = self.conv_up20(torch.cat([layer20_up, layer20], dim=1))
        layer10_up = self.conv_up2(torch.cat([layer2_up, layer2], dim=1))
        layer1_up = self.conv_up10(torch.cat([layer10_up, layer10], dim=1))
        layer0_up = self.conv_up1(torch.cat([layer1_up, layer1], dim=1))
        layer00_up = self.conv_up0(torch.cat([layer0_up, layer0], dim=1))

        layer00_up = torch.cat([layer00_up, layer00, input_tensor], dim=1)
        full_resolution = self.conv_up00(layer00_up)
        full_resolution = torch.cat([full_resolution, input_tensor], dim=1)
        return self.conv_up000(full_resolution)


# Keep the paper-style spelling available without duplicating an implementation.
RadioUNet_C = RadioUNetC
