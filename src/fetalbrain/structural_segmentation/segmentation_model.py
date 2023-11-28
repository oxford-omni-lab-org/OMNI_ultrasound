import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union


class _DoubleConv(nn.Module):
    """Convolutional layer block

    Network block containing two convolutional layers, each followed by batch
    normalization and ReLU activation.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Args:
            in_channels: no. channels of input tensor
            out_channels: no. channels of output tensor
        """
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape [B, C_in, H, W, D]

        Returns:
            output tensor of shape [B, C_out, H, W, D]
        """
        return self.double_conv(x)


class _Down(nn.Module):
    """Downscaling layer block for encoder of unet

    Network block containing a maxpooling layer followed by a double convolutional block.
    The maxpooling layer reduces the spatial dimensions by a factor of 2, and the double conv
    block changes the channel dimension from in_channels to out_channels.

    Optionally a dropout layer is added after the maxpooling layer with a ratio of 0.3.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: bool = False) -> None:
        """
        Args:
            in_channels: channel dimension of input tensor
            out_channels: channel dimension of output tensor
            dropout: whether dropout is applied. Defaults to False.
        """
        super().__init__()
        if dropout:
            self.maxpool_conv = nn.Sequential(nn.MaxPool3d(2), nn.Dropout(0.3), _DoubleConv(in_channels, out_channels))
        else:
            self.maxpool_conv = nn.Sequential(nn.MaxPool3d(2), _DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of size [B, C_in, H_in, W_in, D_in]

        Returns:
            output tensor of size [B, C_out, H_in/2, W_in/2, D_in/2]
        """
        return self.maxpool_conv(x)


class _Up(nn.Module):
    """Upscaling layer block for decoder of unet containing upsampling and convolutional layers.

    Network block that upsamples the input tensor coming from the decoder pathway,
    and then concatenatas this with the output of the corresponding encoder block
    (through the skip connection). The concatenated tensor is then passed through a
    double convolutional block.

    The upsampling is performed either with neareast neighbor upsampling or with
    transposed convolutions.

    """

    def __init__(self, in_channels: int, out_channels: int, transposed_conv: bool = False):
        """
        Args:
            in_channels: channel dimension of the concatenated tensor of the skip connection
            and the previous block in the decoder. This corresponds to the channel dimension
            of x1 + x2 (see forward function).
            out_channels: channel dimension of the output tensor of the convolutional layers.
            transposed_conv: whether to use transposed convolutions. Defaults to False.
        """
        super().__init__()

        self.up: Union[nn.Upsample, nn.ConvTranspose3d]
        if not transposed_conv:
            self.up = nn.Upsample(scale_factor=2, mode="nearest")
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = _DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1: tensor coming from previous decoder block of size [B, C_in, H_in, W_in, D_in]
            x2: tensor coming from skip connection of size [B, C_in/2, 2*W_in, 2*D_in]
        Returns:
            :output tensor of size [B, C_out, H_in*2, W_in*2, D_in*2]
        """
        x1 = self.up(x1)

        # pad x1 and x2 to same size
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffZ = x2.size()[4] - x1.size()[4]

        # padding starts with last dimension and then goes forward
        x1 = F.pad(x1, [diffZ // 2, diffZ - diffZ // 2, diffY // 2, diffY - diffY // 2, diffX // 2, diffX - diffX // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class _OutConv(nn.Module):
    """Final convolutional layer bock reducing the channel dimension to the number of classes.

    Network block contains a single convolutional layer with a kernel size of 1.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Args:
            in_channels: channel dimension of the input tensor (C_in)
            out_channels: number of output channels (i.e. no. classes) (C_out)
        """

        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of size [B, C_in, H, W, D]

        Returns:
            : output tensor of size [B, C_out, H, W, D]
        """
        return self.conv(x)


class UNet(nn.Module):
    """3D Unet architecture for multiclass subcortical segmentation, returns the logits of the prediction
    before any last layer activation. In practice, a soft-max activation should be applied to the
    channel dimension of the logits to obtain a multi-class prediction.

    Example:
        >>> input_im = torch.rand((1, 1, 160, 160, 160)) * 255
        >>> model = UNet(1, 5, min_featuremaps=16, depth=5)
        >>> output = model(input_im)
        >>> assert output.shape == (1, 5, 160, 160, 160)
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int = 5,
        min_featuremaps: int = 64,
        depth: int = 5,
        transposed_conv: bool = False,
    ):
        """
        Args:
            n_channels: number of channel in the input image
            n_classes: number of output classes in the prediction. Defaults to 5.
            min_featuremaps: number of feature maps in the first encoder block. Defaults to 64.
            depth: depth of the unet architecture. Defaults to 5.
            transposed_conv: whether to use transposed convolutions for upsampling. Defaults to False.
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_channels = n_classes
        self.transposed_conv = transposed_conv

        # determine sizes for input and output of downward path
        in_sizes_down = [1, 2, 4, 8, 16, 32][: depth - 1]
        out_sizes_down = [2, 4, 8, 16, 32, 64][: depth - 1]

        # determine sizes for upward path
        in_sizes_up = [x1 + x2 for (x1, x2) in zip(reversed(in_sizes_down), reversed(out_sizes_down))]
        out_sizes_up = in_sizes_down[::-1]

        # Build model
        self.inc = _DoubleConv(n_channels, min_featuremaps)
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        for i in range(depth - 1):
            self.downs.append(
                _Down(int(min_featuremaps * in_sizes_down[i]), int(min_featuremaps * out_sizes_down[i]), dropout=False)
            )
            self.ups.append(_Up(int(min_featuremaps * in_sizes_up[i]), int(min_featuremaps * out_sizes_up[i])))

        self.outc = _OutConv(min_featuremaps, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor with input image of size [B, C_in, H, W, D] with pixel values between 0 and 255

        Returns:
            logits: tensor with output prediction of size [B, C_out, H, W, D]
        """
        x1 = self.inc(x)
        x_list = [x1]

        # Go through downsampling pathway
        for layer in self.downs:
            x_list.append(layer(x_list[-1]))

        # compute bottleneck layer
        x = self.ups[0](x_list[-1], x_list[-2])

        # compute upsampling pathway
        for i, layer in enumerate(self.ups[1:]):
            x = layer(x, x_list[-i - 3])

        logits = self.outc(x)
        return logits
