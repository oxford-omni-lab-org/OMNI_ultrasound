import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Union
from torch.nn import ConvTranspose3d, ConvTranspose2d
from torch.nn import Conv2d, Conv3d


class ConvBlock(nn.Module):
    """
    Convolution Block:
    2x Conv,Norm,RelU
    -dropout
    """

    def __init__(self) -> None:
        super(ConvBlock, self).__init__()

    def _block(self, in_channels: int, features: int, dims: int, name: str, dropout: bool = False) -> nn.Sequential:
        # INSIDE THE BLOCKS, 2 conv with batch norm and relu between

        # Get the different dimensions:
        assert dims in [2, 3]
        instancenorm_layer: Union[type[nn.InstanceNorm2d], type[nn.InstanceNorm3d]]
        conv_layer: Union[type[nn.Conv2d], type[nn.Conv3d]]
        if dims == 3:
            instancenorm_layer = nn.InstanceNorm3d
            conv_layer = nn.Conv3d
        else:
            instancenorm_layer = nn.InstanceNorm2d
            conv_layer = nn.Conv2d

        layers: list[tuple[str, nn.Module]] = [
            (
                name + "conv_1",
                conv_layer(in_channels=in_channels, out_channels=features, kernel_size=3, padding=1, bias=False),
            ),
            # (name + "bnorm_1", nn.BatchNorm3d(num_features=features)),
            (name + "Inorm_1", instancenorm_layer(num_features=features)),
            (name + "relu_1", nn.ReLU(inplace=True)),
            (
                name + "conv_2",
                conv_layer(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False),
            ),
            # (name + "bnorm_2", nn.BatchNorm3d(num_features=features)),
            (name + "Inorm_2", instancenorm_layer(num_features=features)),
            (name + "relu_2", nn.ReLU(inplace=True)),
        ]
        if dropout:
            layers.append((name + "DropOut", nn.Dropout(0.2)))  # DropOut chance

        return nn.Sequential(OrderedDict(layers))


class EncoderBranch(nn.Module):

    """
    Encoder Branch:
    in_channels = #inputs into the network
    features = #features map in the first layer [default =6]
    depth = how deep the network goes [default =4]
    """

    def __init__(self, in_channels: int, features: int = 6, ndims: int = 3, net_depth: int = 4, dropout: bool = False):
        super(EncoderBranch, self).__init__()

        # 2D to 3D:
        assert ndims in [2, 3]
        maxpool_layer: Union[type[nn.MaxPool2d], type[nn.MaxPool3d]]
        if ndims == 3:
            maxpool_layer = nn.MaxPool3d
        else:
            maxpool_layer = nn.MaxPool2d

        self.dims = ndims
        self.depth = net_depth

        # Encoder:
        self.encoder1 = ConvBlock()._block(in_channels, features, ndims, name="encoder_1", dropout=dropout)
        self.pool1 = maxpool_layer(kernel_size=2, stride=2)

        self.encoder2 = ConvBlock()._block(features, features * 2, ndims, name="encoder_2", dropout=dropout)
        self.pool2 = maxpool_layer(kernel_size=2, stride=2)

        self.encoder3 = ConvBlock()._block(features * 2, features * 4, ndims, name="encoder_3", dropout=dropout)
        self.pool3 = maxpool_layer(kernel_size=2, stride=2)

        self.encoder4 = ConvBlock()._block(features * 4, features * 8, ndims, name="encoder_4", dropout=dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # INPUTS INTO ENCODER:
        # Load out
        enc1 = self.encoder1(x)

        assert self.depth in [1, 2, 3, 4]

        if self.depth == 1:
            return enc1
        enc2 = self.encoder2(self.pool1(enc1))
        if self.depth == 2:
            return enc1, enc2
        enc3 = self.encoder3(self.pool2(enc2))
        if self.depth == 3:
            return enc1, enc2, enc3
        enc4 = self.encoder4(self.pool3(enc3))
        return enc1, enc2, enc3, enc4


class BottleNeck(nn.Module):
    def __init__(self, features: int = 6, ndims: int = 3, net_depth: int = 4, dropout: bool = False):
        super(BottleNeck, self).__init__()
        assert ndims in [2, 3]
        maxpool_layer: Union[type[nn.MaxPool2d], type[nn.MaxPool3d]]
        if ndims == 3:
            maxpool_layer = nn.MaxPool3d
        else:
            maxpool_layer = nn.MaxPool3d

        self.pool = maxpool_layer(kernel_size=2, stride=2)
        f = (2 ** (net_depth - 1)) * features
        self.bottleneck = ConvBlock()._block(f, f, ndims, name="Bottleneck", dropout=dropout)

    def forward(self, enc_out: torch.Tensor) -> torch.Tensor:
        """
        enc_out = enc_output[-1] the last ouput of the encoder
        """
        return self.bottleneck(self.pool(enc_out))


class DecoderBranch(nn.Module):

    """
    dec_depth == 1 (the level at which the decoder ouputs something)(old fine_tune_level)
                DEFAULT =1 (back to the orginal layer)
    """

    def __init__(
        self, features: int = 6, ndims: int = 3, net_depth: int = 4, dec_depth: int = 1, dropout: bool = False
    ) -> None:
        super(DecoderBranch, self).__init__()

        self.dec_depth = dec_depth
        self.net_depth = net_depth

        assert ndims in [2, 3]
        convtrans_layer: Union[type[ConvTranspose3d], type[ConvTranspose2d]]
        if ndims == 3:
            convtrans_layer = ConvTranspose3d
        else:
            convtrans_layer = ConvTranspose2d

        assert self.dec_depth <= self.net_depth  # assert the depth of the deocder is less than the encoder depth

        self.upconv1 = convtrans_layer(
            features * 8, features * 8, kernel_size=2, stride=2
        )  # the lowest of the branches
        self.decoder1 = ConvBlock()._block((features * 8) * 2, features * 4, ndims, name="decoder_4", dropout=dropout)

        self.upconv2 = convtrans_layer(features * 4, features * 4, kernel_size=2, stride=2)
        self.decoder2 = ConvBlock()._block((features * 4) * 2, features * 2, ndims, name="decoder_3", dropout=dropout)

        if self.dec_depth < 3:
            self.upconv3 = convtrans_layer(features * 2, features * 2, kernel_size=2, stride=2)
            self.decoder3 = ConvBlock()._block((features * 2) * 2, features, ndims, name="decoder_3", dropout=dropout)

        if self.dec_depth < 2:
            self.upconv4 = convtrans_layer(features, features, kernel_size=2, stride=2)
            self.decoder4 = ConvBlock()._block((features) * 2, features, ndims, name="decoder_1", dropout=dropout)

    def forward(self, bottleneck: torch.Tensor, enc_output: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        bottleneck - first input into the network
        enc_output - a list of the enc outputs to put back in e.g. enc_output = [enc1,enc2,enc3...]
        """

        # DEPTH (net_depth) OF THE NETWORK

        # first layer block
        start_conv = 5 - self.net_depth
        dec = getattr(self, f"upconv{start_conv}")(bottleneck)
        dec = torch.cat((dec, enc_output[-1]), dim=1)
        dec = getattr(self, f"decoder{start_conv}")(dec)
        if self.dec_depth == self.net_depth:
            return dec

        # loop through the other layer blocks
        for i in range(1, self.net_depth):
            dec = getattr(self, f"upconv{start_conv + i}")(dec)
            dec = torch.cat((dec, enc_output[-(i + 1)]), dim=1)
            dec = getattr(self, f"decoder{start_conv + i}")(dec)

            if self.dec_depth == (self.net_depth - i):
                return dec

        raise ValueError("dec_depth must be less than or equal to net_depth")


class UNet_MW(nn.Module):
    """
    My U-Net
    inputs:
    in_channels = number of channels in the input
    out_channels = number of output channel
    features = initial numnber of features
    dec_depth = the depth that the output comes from (1 back to the top)
    net__depth = how deep the network is
    """

    def __init__(self) -> None:
        """
        Set up the U-Net
        """

        super(UNet_MW, self).__init__()
        in_channels = 1
        out_channels = 1
        features = 6
        net_depth = 4
        dropout = True
        dec_depth = 1
        ndims = 3
        self.output_activation = "sigmoid"

        # --------- 1. Encoder:
        self.enc = EncoderBranch(in_channels, features, ndims, net_depth, dropout)
        # --------- 2. Bottleneck:
        self.bottleneck = BottleNeck(features, ndims, net_depth, dropout)
        # --------- 3. Decoder:
        self.dec = DecoderBranch(features, ndims, net_depth, dec_depth, dropout)

        self.conv: Union[Conv2d, Conv3d]
        if ndims == 3:
            self.conv = Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)
        elif ndims == 2:
            self.conv = Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)
        else:
            raise ValueError("ndims must be 2 or 3")

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        enc_outputs = self.enc(x)
        BottleNeck = self.bottleneck(enc_outputs[-1])
        dec_output = self.dec(BottleNeck, enc_outputs)
        cnn_output = self.conv(dec_output)

        if self.output_activation == "sigmoid":
            return torch.sigmoid(cnn_output)
        else:
            raise NotImplementedError("Only sigmoid activation is currently supported")
