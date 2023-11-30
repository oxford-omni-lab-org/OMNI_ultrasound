import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from torch.distributions.normal import Normal
from torch.nn import Conv2d, Conv3d
from fetalbrain.tedsnet_multi.network.UNet import DecoderBranch


class WholeDiffeoUnit(nn.Module):
    """
    The diffeo block:
    1. computes the decoder branch
    2. generates an ndims field
    3. applies diffeo intergretion
    4. applies field to prior shape
    """

    def __init__(self) -> None:
        super(WholeDiffeoUnit, self).__init__()

        # Get parameters from params:
        self.out_channels = 10
        self.ndims = 3
        self.viscous = 1
        self.act = 1
        self.features = 6
        self.dropout = 1
        self.net_depth = 4
        self.dec_depth = 1
        self.inshape = [160, 160, 160]
        self.int_steps = 8
        self.Guas_kernel = 5
        self.Guas_P = 2.0
        self.mega_P = 1

        # GET DECODER OUTPUT:
        # happy with this
        self.dec = DecoderBranch(
            features=self.features,
            ndims=self.ndims,
            net_depth=self.net_depth,
            dec_depth=self.dec_depth,
            dropout=self.dropout,
        )

        # Size of initial flow field:
        frac_size_change = [1, 2, 4, 8]  # The fractional change
        self.flow_field_size = [int(s / frac_size_change[self.dec_depth - 1]) for s in self.inshape]
        # Size of upsampled flow field:
        self.Mega_inshape = [s * self.mega_P for s in self.inshape]

        # 1. GENERATE FIELDS
        self.gen_field = GenDisField(self.dec_depth, self.features, self.ndims)

        # 2. Apply diffeomorphic settings :
        self.diffeo_field = DiffeoUnit(
            self.flow_field_size, self.Mega_inshape, self.int_steps, self.Guas_kernel, self.Guas_P
        )

        # 3.  Apply transform to prior:
        self.transformer = mw_SpatialTransformer(self.Mega_inshape)

    def forward(
        self,
        BottleNeck: torch.Tensor,
        enc_outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        prior_shape: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get decoder:
        dec_output = self.dec(BottleNeck, enc_outputs)
        flow_field = self.gen_field(dec_output)
        flow_upsamp = self.diffeo_field(flow_field, self.act, self.viscous)
        sampled = WarpPriorShape(self, prior_shape, flow_upsamp)

        return flow_field, flow_upsamp, sampled


class GenDisField(nn.Module):
    """
    From the output of the U-Net generate the correct sized fields

    input: dec output [batch,feature maps,...]
    output: flow_Field [batch,ndims,...], with inialised wieghts
    """

    def __init__(self, layer_nb: int, features: int, ndims: int):
        super().__init__()

        self.conv_layer: Union[type[torch.nn.Conv2d], type[torch.nn.Conv3d]]
        if ndims == 3:
            self.conv_layer = Conv3d
        elif ndims == 2:
            self.conv_layer = Conv2d

        dec_features = [1, 1, 2, 4]  # number of features from each decoder level
        self.flow_field = self.conv_layer(
            dec_features[layer_nb - 1] * features, out_channels=ndims, kernel_size=1
        )  # Out_channels =3 (x,y,z), could be kerne=3 (??)
        self.flow_field.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow_field.weight.shape))
        if self.flow_field.bias is not None:
            self.flow_field.bias = nn.Parameter(torch.zeros(self.flow_field.bias.shape))

    def forward(self, CNN_output: torch.Tensor) -> torch.Tensor:
        return self.flow_field(CNN_output)


class DiffeoUnit(nn.Module):
    """
    Takes in an initial field and ouputs the final upsampled field:

    Takes in an [ndim,M,N] array which acts as the field
    1. Activation Func:
    2. Amplify across integration layers:
    3. Super upsample to quantiity needed:
    """

    def __init__(
        self,
        flow_field_size: list[int],
        mega_size: list[int],
        int_steps: int = 7,
        Guas_kernel: int = 5,
        Guas_P: float = 2.0,
    ):
        super(DiffeoUnit, self).__init__()

        # --  1. Intergration Layers:
        self.flow_field_size = flow_field_size
        self.integrate_layer = mw_DiffeoLayer(flow_field_size, int_steps, Guas_kernel, Guas_P=Guas_P)

        # -- 2. Mega Upsample:
        self.Mega_inshape = tuple(mega_size)
        modes = {2: "bilinear", 3: "trilinear"}
        self.MEGAsmoothing_upsample = nn.Upsample(
            self.Mega_inshape, mode=modes[len(flow_field_size)], align_corners=False
        )

    def forward(self, flow_field: torch.Tensor, act: int, viscous: bool) -> torch.Tensor:
        # 1. Activation Func = between the required amounts:
        if act:
            flow_field = DiffeoActivat(flow_field, self.flow_field_size)

        # 2. Get the displacment field:
        amplified_flow_field = self.integrate_layer(flow_field, viscous)

        # 3. Super Upsample:
        flow_upsamp = self.MEGAsmoothing_upsample(amplified_flow_field)  # Upsample

        return flow_upsamp


class mw_DiffeoLayer(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    Adapted from: https://github.com/voxelmorph/voxelmorph

    """

    def __init__(self, inshape: list[int], nsteps: int, kernel: int = 3, Guas_P: float = 2.0):
        super().__init__()

        assert nsteps >= 0, "nsteps should be >= 0, found: %d" % nsteps
        self.nsteps = nsteps
        # Set up spatial transformer to intergrate the flow field:
        self.transformer = mw_SpatialTransformer(inshape)

        # ------------------------------
        # SMOOTHING KERNEL:
        # ------------------------------
        ndims = len(inshape)
        self.sigma = Guas_P
        self.SmthKernel = GaussianSmoothing(channels=ndims, kernel_size=kernel, sigma=Guas_P, dim=ndims)
        # ------------------------------
        # ------------------------------

    def forward(self, vec: torch.Tensor, viscous: bool = True) -> torch.Tensor:
        for n in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
            if viscous:
                # if viscous methods, then smooth at each composition:
                vec = self.SmthKernel(vec)

        return vec


class mw_SpatialTransformer(nn.Module):
    """
    The pytorch spatial Transformer

    Pytorch transformers require grids generated between -1---1.
    src - Prior shape or flow field [2,3,x,x,x]
    flow -
    """

    def __init__(self, size: list[int], mode: str = "bilinear"):
        super().__init__()

        self.mode = mode
        # create sampling grid (in Pytorch terms)
        vectors = [torch.linspace(-1, 1, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.float()
        self.register_buffer("grid", grid)  # not trained by the optimizer, saves memory

    def forward(self, src: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # Pytorch requires axis switch:
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]

        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True)


class GaussianSmoothing(nn.Module):
    """
    Adrian Sahlman:
    https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/7
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel. If it is less than 0, then it will
            learn sigma
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(
        self,
        channels: int,
        kernel_size: Union[int, list[int]] = 5,
        sigma: Union[float, list[float]] = 2.0,
        dim: int = 2,
    ):
        super(GaussianSmoothing, self).__init__()
        # sigma =2
        self.og_sigma = sigma

        if isinstance(kernel_size, int):
            kernel_list = tuple([kernel_size] * dim)
        else:
            kernel_list = tuple(kernel_size)

        if isinstance(sigma, float):
            sigma = [sigma] * dim

        kernel_dic = {3: 1, 5: 2}
        self.pad = kernel_dic[kernel_list[0]]

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_list])
        kernel = torch.ones_like(meshgrids[0])

        for size, std, mgrid in zip(kernel_list, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp((-(((mgrid - mean) / std) ** 2)) / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        if sigma[0] < 0:
            # --- Learnable Sigma---------------:
            sigma = 2
            self.learnable = 1  # is the network learnable or static?

            self.conv: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d]
            if dim == 1:
                assert len(kernel_list) == 1
                self.conv = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=kernel_list, padding=self.pad)
            elif dim == 2:
                assert len(kernel_list) == 2
                self.conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_list, padding=self.pad)
            elif dim == 3:
                assert len(kernel_list) == 3
                self.conv = nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=kernel_list, padding=self.pad)
            else:
                raise RuntimeError("Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim))

            # Initialse with normal dist
            self.conv.weight = nn.Parameter(torch.cat((kernel, kernel), dim=1))
            if self.conv.bias is not None:
                self.conv.bias = nn.Parameter(torch.zeros(self.conv.bias.shape))

        else:
            # --- Static network---------------:
            self.learnable = 0

            self.register_buffer("weight", kernel)

            self.groups = channels

            if dim == 1:
                self.conv = F.conv1d  # type: ignore
            elif dim == 2:
                self.conv = F.conv2d   # type: ignore
            elif dim == 3:
                self.conv = F.conv3d  # type: ignore
            else:
                raise RuntimeError("Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        # if static or trainable:
        if self.learnable == 1:
            return self.conv(input)
        else:
            return self.conv(input, weight=self.weight, groups=self.groups, padding=self.pad)


def DiffeoActivat(flow_field: torch.Tensor, size: list[int]) -> torch.Tensor:
    """Activation Function

    Args:
        flow_field ([tensor array]): A n-dimension array containing the flow feild in each direction
        size ([list]): [description]: The maximum size of the field, to limit the size of the intial displacement

    Returns:
        flow_field [tensor array]: Flow field after the activation funciton has been applied.
    """

    # Assert ndims is 2D or 3D
    assert flow_field.size()[1] in [2, 3]
    assert len(size) in [2, 3]

    if len(size) == 3:
        flow_1 = torch.tanh(flow_field[:, 0, :, :, :]) * (1 / size[0])
        flow_2 = torch.tanh(flow_field[:, 1, :, :, :]) * (1 / size[1])
        flow_3 = torch.tanh(flow_field[:, 2, :, :, :]) * (1 / size[2])
        flow_field = torch.stack((flow_1, flow_2, flow_3), dim=1)
    elif len(size) == 2:
        flow_1 = torch.tanh(flow_field[:, 0, :, :]) * (1 / size[0])
        flow_2 = torch.tanh(flow_field[:, 1, :, :]) * (1 / size[1])
        flow_field = torch.stack((flow_1, flow_2), dim=1)

    return flow_field


def WarpPriorShape(diffeo_unit: WholeDiffeoUnit, prior_shape: torch.Tensor, disp_field: torch.Tensor) -> torch.Tensor:
    """
    Tranform a set of prior shapes:
    """
    # Apply displacment field
    disp_prior_shape = diffeo_unit.transformer(prior_shape, disp_field)

    return disp_prior_shape
