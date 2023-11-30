import torch.nn as nn
import torch
from typing import Union
from fetalbrain.tedsnet_multi.network.UNet import EncoderBranch, BottleNeck
from fetalbrain.tedsnet_multi.network.TEDS_elements import WholeDiffeoUnit


class TEDS_Net(nn.Module):
    """
    TEDS-Net:

    Input is the parameter describing dictionary.

    """

    def __init__(self) -> None:
        super(TEDS_Net, self).__init__()

        # Parameters settings - arch:
        in_channels = 1
        features = 6
        net_depth = 4
        dropout = 1
        self.no_branches = 1

        # Parameters settings - diffeomorphic:
        self.mega_P = 1

        # Dataset dependant parameters
        ndims = 3

        # -------------------------------------------------------------------
        # --------- 1. Enc:
        self.enc = EncoderBranch(in_channels, features, ndims, net_depth, dropout)

        # --------- 2. Bottleneck:
        self.bottleneck = BottleNeck(features, ndims, net_depth, dropout)

        # --------- 3. Decoder + Diffeo Units:
        self.STN = WholeDiffeoUnit()

        # --------------------------------------------------------------------
        # --------- 4. Downsample to up-sampled fields (visualisation):
        if self.mega_P > 1:
            max_pool_layer: Union[type[nn.MaxPool2d], type[nn.MaxPool3d]]
            if ndims == 2:
                max_pool_layer = nn.MaxPool2d
            elif ndims == 3:
                max_pool_layer = nn.MaxPool3d

            # downsample the final results
            self.downsample = max_pool_layer(kernel_size=3, stride=self.mega_P, padding=1)

    def forward(self, x: torch.Tensor, prior_shape: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            x: input tensor of shape [B, C_in, H, W, D]
            prior_shape: prior shape tensor of shape [B, C_num, H, W, D], containing a prior shape
                for each of the classes (C_num)

        Returns:
            sampled: sampled tensor of shape [B, C_num, H, W, D]
            flow_upsamp: tensor of shape [B, 3, H, W, D], containing the upsampled flow fields

        """

        # -------- 1. Enc + Bottleneck:
        enc_outputs = self.enc(x)
        BottleNeck = self.bottleneck(enc_outputs[-1])

        # --------- 2. Dec + Diffeo:
        _, flow_upsamp, sampled = self.STN(BottleNeck, enc_outputs, prior_shape)
        # DOWNSAMPLE
        if self.mega_P > 1:
            sampled = self.downsample(sampled)

        return sampled, flow_upsamp
