"""
This module contains the architecture of the alignment model fBAN_v1 that aligns the input scan
to a reference coordinate system. The rotations in this model are represented by quaternions.
The model consists of predicting coarse and fine alignment parameters after each other.

More details about the architecture can be found in the NeuroImage paper
(`Moser et al. NeuroImage, 2022) <https://www.sciencedirect.com/science/article/pii/S1053811922004608>`_).

[Local] Network code copied from the OMNI cluster: **../../../shared/stru0039/fBAN/v1/model.py**

"""
import torch
import torch.nn as nn
import torch.nn.functional as ff
from .kelluwen_transforms import (
    generate_affine,
    apply_affine,
    deconstruct_affine,
)


class AlignmentModel(nn.Module):
    """Model class for the alignment model fBAN which predicts the alignment
    parameters for a given scan.

    Args:
        shape_data: the input shape of the image
        dimensions_hidden: the dimensions of the hidden layers in the network
        size_kernel: convolutional kernel size, defaults to 3

    :Example:
        >>> model = AlignmentModel()
    """

    def __init__(
        self,
        shape_data: list[int] = [1, 160, 160, 160],
        dimensions_hidden: list[int] = [16, 32, 64, 128, 256, 128, 64, 32],
        size_kernel: int = 3,
    ) -> None:
        super(AlignmentModel, self).__init__()

        # Rename for readability
        dh = dimensions_hidden
        sd = shape_data
        sk = size_kernel

        # Convolutional blocks X1
        self.enc_X1_cb00 = self._conv_block(sd[0], dh[0], sk)
        self.enc_X1_cb01 = self._conv_block(dh[0], dh[1], sk)
        self.enc_X1_cb02 = self._conv_block(dh[1], dh[2], sk)
        self.enc_X1_cb03 = self._conv_block(dh[2], dh[3], sk)
        self.enc_X1_cb04 = self._conv_block(dh[3], dh[4], sk)

        # Dense layers X1
        self.dense_X1_00 = nn.Linear(dh[4] * 10 * 10 * 10, 8)

        # Convolutional blocks X2
        self.enc_X2_cb00 = self._conv_block(sd[0], dh[0], sk)
        self.enc_X2_cb01 = self._conv_block(dh[0], dh[1], sk)
        self.enc_X2_cb02 = self._conv_block(dh[1], dh[2], sk)
        self.enc_X2_cb03 = self._conv_block(dh[2], dh[3], sk)
        self.enc_X2_cb04 = self._conv_block(dh[3], dh[4], sk)

        # Dense layers X2
        self.dense_X2_00 = nn.Linear(dh[4] * 10 * 10 * 10, 8)

        # Deterministic
        self.maxpool = nn.MaxPool3d(2, 2)

    def _conv_block(self, ch_in: int, ch_out: int, size_kernel: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv3d(ch_in, ch_out, size_kernel, padding="same"),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(),
            nn.Conv3d(ch_out, ch_out, size_kernel, padding="same"),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the alignment model

        Args:
            x: batch of data of size [B, 1, H, W, D] with pixel values between 0 and 255

        Returns:
            translation: translation parameters of size [B, 3] between 0 and 1
            rotation: rotation parameters of size [B, 4], represented as quarternions
            scaling: scaling parameters of size [B, 3] between 0 and 1

        :Example:
            >>> model = AlignmentModel()
            >>> dummy_scan = torch.rand(1,1,160,160,160)
            >>> translation, rotation, scaling = model(dummy_scan)
        """
        # Normalize input to [0,1]
        x = x / 255

        # Model X1
        hidden = self.enc_X1_cb00(x)
        hidden = self.enc_X1_cb01(self.maxpool(hidden))
        hidden = self.enc_X1_cb02(self.maxpool(hidden))
        hidden = self.enc_X1_cb03(self.maxpool(hidden))
        hidden = self.enc_X1_cb04(self.maxpool(hidden))
        hidden = self.dense_X1_00(torch.flatten(hidden, start_dim=1))

        # Parameters X1
        translation_X1 = hidden[:, :3]  # 3 translation parameter
        rotation_X1 = ff.normalize(hidden[:, 3:7])  # 4 rotation parameters
        scaling_X1 = hidden[:, 7:8]  # 1 scaling parameter

        # Transform X1
        transform_X1 = generate_affine(
            parameter_translation=translation_X1 * 160,  # Scan shape is (160,160,160)
            parameter_rotation=rotation_X1,
            parameter_scaling=scaling_X1.tile(1, 3),
            type_rotation="quaternions",
            transform_order="srt",
        )

        # Scan aligned X1
        x_aligned_X1 = apply_affine(
            image=x,
            transform_affine=transform_X1,
            type_resampling="bilinear",
            type_origin="centre",
        )

        # Model X2
        hidden = self.enc_X2_cb00(x_aligned_X1)
        hidden = self.enc_X2_cb01(self.maxpool(hidden))
        hidden = self.enc_X2_cb02(self.maxpool(hidden))
        hidden = self.enc_X2_cb03(self.maxpool(hidden))
        hidden = self.enc_X2_cb04(self.maxpool(hidden))
        hidden = self.dense_X2_00(torch.flatten(hidden, start_dim=1))

        # Parameters X2
        translation_X2 = hidden[:, :3]  # 3 translation parameter
        rotation_X2 = ff.normalize(hidden[:, 3:7])  # 4 rotation parameters
        scaling_X2 = hidden[:, 7:8]  # 1 scaling parameter

        # Transform X2
        transform_X2 = generate_affine(
            parameter_translation=translation_X2 * 160,  # Scan shape is (160,160,160)
            parameter_rotation=rotation_X2,
            parameter_scaling=scaling_X2.tile(1, 3),
            type_rotation="quaternions",
            transform_order="srt",
        )

        # Combined transform
        transform = transform_X2 @ transform_X1

        # Predicted parameters
        translation, rotation, scaling = deconstruct_affine(
            transform_affine=transform,
            transform_order="srt",
            type_rotation="quaternions",
        )

        assert isinstance(translation, torch.Tensor)
        assert isinstance(rotation, torch.Tensor)
        assert isinstance(scaling, torch.Tensor)

        return translation / 160, rotation, scaling
