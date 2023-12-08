"""
This module contains helper functions to apply alignment parameters resulting from fBAN to an image. It is likely not
necessary to access these functions directly when doing standard image alignment operations, as most are shadowed
by functions in the main align.py module.

These functions have been taken from the
`Kelluwen Github <https://github.com/FelipeMoser/kelluwen/blob/main/kelluwen/functions/transforms.py>`_.

The functions were copied over to minimise dependendencies, and their style adjusted to match the rest of the codebase.

Module functions
----------------

"""

from typing import Union, overload, Literal, Optional, TypedDict
import torch
from typeguard import typechecked


ORDER_TYPES = Literal["trs", "tsr", "rts", "rst", "str", "srt"]
ROT_TYPES = Literal["euler_xyz", "euler_xzy", "euler_yxz", "euler_yzx", "euler_zxy", "euler_zyx", "quaternions"]
RETURN_TYPES = Literal["positional", "named"]
ORIGIN_TYPES = Literal["centre", "origin"]


@typechecked
def deconstruct_affine(
    transform_affine: torch.Tensor,
    transform_order: ORDER_TYPES = "srt",
    type_rotation: ROT_TYPES = "euler_xyz",
    type_output: RETURN_TYPES = "positional",
) -> Union[tuple, dict[str, torch.Tensor]]:
    """Deconstructs the affine transform into its conforming translation, rotation, and scaling parameters.

    Args:
        transform_affine: Affine transform being deconstructed. Must be of shape (B, C, H, W, D)
            or (B, H, W, D).
        transform_order : Order of multiplication of translation, rotation, and scaling transforms, defaults to 'srt'
        type_rotation: Type of rotation parameters: quaternions or Euler angles. For Euler angles, the order of the
            multiplication of the rotations around x, y, and z is represented in the name (euler_xyz, euler_yzx, etc.),
            defaults to 'euler_xyz'
        type_output : Determines how the outputs are returned. If set to positional, it returns positional outputs.
            If set to named, it returns a dictionary with named outputs, defaults to 'positional'

    Returns:
        parameter_translation: tensor of size (B, C, 3) (channel dimension is optional, based on whether channel
            dimension is present in input affine)
        parameter_rotation : tensor of size (B, C, 3) (if type_rotation is euler) or
            (B, C,  4) (if type_rotation is 'quaternions')
        parameter_scaling : tensor of size [B, C, 3]

    Example:
        >>> transform_affine = torch.eye(4,4, dtype=torch.float32).unsqueeze(0)
        >>> transl, rot, scale = deconstruct_affine(transform_affine, transform_order='srt', type_rotation='euler_xyz',\
            type_output='positional')

    """

    # Validate arguments
    if transform_affine.dim() not in (3, 4):
        raise ValueError(f"expected a 3D or 4D transform_affine, got {transform_affine.dim()!r}D instead")
    if transform_affine.shape[-2:] not in ((3, 3), (4, 4)):
        raise ValueError(f"unexpected shape of transform_affine {transform_affine.shape!r}")

    # Update variables if required
    if transform_affine.dim() == 4:
        channel_dimension = True
    elif transform_affine.dim() == 3:
        channel_dimension = False
        transform_affine = transform_affine[:, None, ...]

    # Extract scaling parameters
    if transform_order in ("srt, str, tsr"):
        parameter_scaling = transform_affine[..., :-1, :-1].norm(dim=3)
    else:
        parameter_scaling = transform_affine[..., :-1, :-1].norm(dim=2)

    # Extract scaling transform
    transform_scaling = generate_scaling(parameter_scaling, type_output="positional")
    assert isinstance(transform_scaling, torch.Tensor)

    # Extract rotation transform
    if transform_order in ("srt, str, tsr"):
        transform_rotation = transform_scaling.inverse() @ transform_affine
    else:
        transform_rotation = transform_affine @ transform_scaling.inverse()
    transform_rotation[..., :-1, -1] = 0

    # Extract translation transform
    if transform_order in ("trs", "tsr"):
        transform_translation = torch.eye(4).tile((*transform_affine.shape[:2], 1, 1))
        transform_translation[..., :-1, -1] = transform_affine[..., :-1, -1]
    elif transform_order == "str":
        transform_translation = torch.eye(4).tile((*transform_affine.shape[:2], 1, 1))
        transform_translation[..., :-1, -1] = (transform_scaling.inverse() @ transform_affine)[..., :-1, -1]
    elif transform_order == "rts":
        transform_translation = transform_rotation.inverse() @ transform_affine @ transform_scaling.inverse()
    elif transform_order == "rst":
        transform_translation = transform_scaling.inverse() @ transform_rotation.inverse() @ transform_affine
    elif transform_order == "srt":
        transform_translation = transform_rotation.inverse() @ transform_scaling.inverse() @ transform_affine

    # Extract translation parameters
    parameter_translation = transform_translation[..., :-1, -1]

    # Extract rotation parameters
    if transform_rotation.shape[2] == 2:  # 2D rotation
        parameter_rotation = torch.asin(transform_rotation[..., 1, 0])
    else:  # 3D rotation
        if type_rotation == "quaternions":
            # Extract quaternions from rotation transform
            # This section has been adapted to pytorch from nibabel's implementation:
            # htorchps://nipy.org/nibabel/reference/nibabel.quaternions.html
            transform_rotation = transform_rotation[..., :-1, :-1].flatten(start_dim=-2)
            Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz = [
                transform_rotation[..., i] for i in range(transform_rotation.shape[-1])
            ]

            K = torch.eye(4).tile(*transform_rotation.shape[:-1], 1, 1)
            K[..., 0, 0] = Qxx - Qyy - Qzz
            K[..., 1, 0] = Qyx + Qxy
            K[..., 1, 1] = Qyy - Qxx - Qzz
            K[..., 2, 0] = Qzx + Qxz
            K[..., 2, 1] = Qzy + Qyz
            K[..., 2, 2] = Qzz - Qxx - Qyy
            K[..., 3, 0] = Qyz - Qzy
            K[..., 3, 1] = Qzx - Qxz
            K[..., 3, 2] = Qxy - Qyx
            K[..., 3, 3] = Qxx + Qyy + Qzz
            K /= 3
            vals, vecs = torch.linalg.eigh(K)
            q = vecs[..., [3, 0, 1, 2], :]
            parameter_rotation = torch.zeros([*transform_rotation.shape[:2], 4], device=transform_affine.device)
            idx = torch.argmax(vals, dim=-1)
            for i in range(q.shape[0]):
                for j in range(q.shape[1]):
                    parameter_rotation[i, j] = q[i, j, :, idx[i, j]]
                    if parameter_rotation[i, j, 0] < 0:
                        parameter_rotation[i, j] *= -1
        else:
            # Get indices for the necessary transform components
            EulerDict = TypedDict(
                "EulerDict", {"s": int, "alpha": list[list[int]], "beta": list[int], "gamma": list[list[int]]}
            )
            euler_idx: dict[str, EulerDict] = dict(
                euler_xyz=dict(s=-1, alpha=[[1, 2], [2, 2]], beta=[0, 2], gamma=[[0, 1], [0, 0]]),
                euler_xzy=dict(s=1, alpha=[[2, 1], [1, 1]], beta=[0, 1], gamma=[[0, 2], [0, 0]]),
                euler_yxz=dict(s=1, alpha=[[0, 2], [2, 2]], beta=[1, 2], gamma=[[1, 0], [1, 1]]),
                euler_yzx=dict(s=-1, alpha=[[2, 0], [0, 0]], beta=[1, 0], gamma=[[1, 2], [1, 1]]),
                euler_zxy=dict(s=-1, alpha=[[0, 1], [1, 1]], beta=[2, 1], gamma=[[2, 0], [2, 2]]),
                euler_zyx=dict(s=1, alpha=[[1, 0], [0, 0]], beta=[2, 0], gamma=[[2, 1], [2, 2]]),
            )
            idx_s = euler_idx[type_rotation]["s"]
            idx_alpha = euler_idx[type_rotation]["alpha"]
            idx_beta = euler_idx[type_rotation]["beta"]
            idx_gamma = euler_idx[type_rotation]["gamma"]

            # Get Euler angles
            alpha = torch.atan2(
                idx_s * transform_rotation[..., idx_alpha[0][0], idx_alpha[0][1]],
                transform_rotation[..., idx_alpha[1][0], idx_alpha[1][1]],
            )
            beta = -idx_s * torch.asin(transform_rotation[..., idx_beta[0], idx_beta[1]])
            gamma = torch.atan2(
                idx_s * transform_rotation[..., idx_gamma[0][0], idx_gamma[0][1]],
                transform_rotation[..., idx_gamma[1][0], idx_gamma[1][1]],
            )
            parameter_rotation = torch.stack([alpha, beta, gamma], dim=2)

    # Remove channels if required
    if channel_dimension is False:
        parameter_scaling = parameter_scaling[:, 0, :]
        parameter_translation = parameter_translation[:, 0, :]
        parameter_rotation = parameter_rotation[:, 0, :]

    # Return results
    if type_output == "positional":
        return parameter_translation, parameter_rotation, parameter_scaling
    else:
        return {
            "parameter_translation": parameter_translation,
            "parameter_rotation": parameter_rotation,
            "parameter_scaling": parameter_scaling,
        }


@typechecked
def apply_affine(
    image: torch.Tensor,
    transform_affine: torch.Tensor,
    shape_output: Optional[Union[torch.Size, list[int]]] = None,
    type_resampling: Literal["bilinear", "nearest"] = "bilinear",
    type_origin: ORIGIN_TYPES = "centre",
    type_output: RETURN_TYPES = "positional",
) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
    """Applies affine transform to tensor.

    Args:
        image: image being transformed. Must be of shape (B, C, `*`).
        transform_affine: affine transform being applied. Must be of shape (B, C, `*`),
            (B, 1, `*`), or (B, `*`).
        shape_output: Output shape of transformed image. Must have the same batch and channel as image.
            If None, the output_shape=image.shape. Defaults to None.
        type_resampling: interpolation algorithm used when sampling image. Available: bilinear, nearest
        type_origin: point around which the transform is applied, defaults to centre
        type_output: Determines how the outputs are returned. If set to positional, it returns positional outputs.
            If set to named, it returns a dictionary with named outputs.

    Returns:
        image_transformed

    Example:
        >>> image = torch.rand((1, 1, 160, 160, 160))
        >>> identity_affine = torch.eye(4, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        >>> image_transformed = apply_affine(image, identity_affine)
    """

    # Validate arguments
    if image.dim() not in (4, 5):
        raise ValueError(f"expected a 4D or 5D image, got {image.dim()!r}D instead")
    if transform_affine.dim() not in (3, 4):
        raise ValueError(f"expected a 3D or 4D transform_affine, got {transform_affine.dim()!r}D instead")
    if transform_affine.shape[0] != image.shape[0]:
        raise ValueError("transform_affine.shape doesn't match image.shape")
    if transform_affine.dim() - 2 == image.dim() - 3:
        if transform_affine.shape[0] != image.shape[0]:
            raise ValueError("transform_affine.shape doesn't match image.shape")
    if transform_affine.shape[-2:] != (*[image.dim() - 1] * 2,):
        raise ValueError("transform_affine.shape doesn't match image.shape")
    if shape_output is not None:
        if image.dim() != len(shape_output) or image.shape[:2] != shape_output[:2]:
            raise ValueError("shape_output doesn't match image.shape")

    if shape_output is None:
        shape_output = image.shape
    if transform_affine.dim() == 3:
        transform_affine = transform_affine[:, None, :, :]
    if image.type() not in (torch.float, torch.double):
        image = image.float()
    if transform_affine.type() != (torch.float, torch.double):
        transform_affine = transform_affine.float()

    # Translate origin if required
    if type_origin == "centre":
        transform_origin = torch.eye(image.dim() - 1).to(transform_affine.device)
        transform_origin = transform_origin.tile(*transform_affine.shape[:2], 1, 1)
        transform_origin[..., :-1, -1] = -(torch.tensor(image.shape[2 : image.dim()]) - 1) / 2
        transform_affine = transform_origin.inverse() @ transform_affine @ transform_origin

    # Generate transformed coordinates
    transform_affine_n = transform_affine.inverse()[..., :-1, :]

    # changed to xy should be ij
    coordinates_seq = torch.meshgrid(*(torch.arange(s) for s in shape_output[2:]), indexing="ij")
    coordinates = torch.stack((*coordinates_seq, torch.ones(*shape_output[2:]))).to(image.device)

    coordinates = transform_affine_n @ (coordinates.reshape((1, 1, image.dim() - 1, -1)))

    # Prepare indices for readability
    batch = torch.arange(shape_output[0])[:, None, None]
    channel = torch.arange(shape_output[1])[None, :, None]
    x = coordinates[..., 0, :]
    y = coordinates[..., 1, :]
    if image.dim() == 5:
        z = coordinates[:, :, 2, :]

    # Find transformed coordinates that lie outside image
    mask = ~(torch.any(coordinates < 0, dim=2) | (x > image.shape[2] - 1) | (y > image.shape[3] - 1))
    if image.dim() == 5:
        mask = mask & ~(z > image.shape[4] - 1)

    # Clip coordinates outside image
    coordinates *= mask[:, :, None, :]

    # Resample
    if type_resampling == "nearest":
        # Prepare indices and weights for readability
        c0 = lambda x: (x.ceil() - 1).long()  # noqa: E731
        c1 = lambda x: x.ceil().long()  # noqa: E731
        w0 = lambda x: x.ceil() - x.round()  # noqa: E731
        w1 = lambda x: x.round() - (x.ceil() - 1)  # noqa: E731

    elif type_resampling == "bilinear":
        # Prepare indices and weights for readability
        c0 = lambda x: (x.ceil() - 1).long()  # noqa: E731
        c1 = lambda x: x.ceil().long()  # noqa: E731
        w0 = lambda x: x.ceil() - x  # noqa: E731
        w1 = lambda x: x - (x.ceil() - 1)  # noqa: E731

    # Sample transformed image
    if image.dim() == 4:
        image_transformed = (
            image[batch, channel, c0(x), c0(y)] * (w0(x) * w0(y))
            + image[batch, channel, c1(x), c0(y)] * (w1(x) * w0(y))
            + image[batch, channel, c0(x), c1(y)] * (w0(x) * w1(y))
            + image[batch, channel, c1(x), c1(y)] * (w1(x) * w1(y))
        )
    else:
        image_transformed = (
            image[batch, channel, c0(x), c0(y), c0(z)] * (w0(x) * w0(y) * w0(z))
            + image[batch, channel, c1(x), c0(y), c0(z)] * (w1(x) * w0(y) * w0(z))
            + image[batch, channel, c0(x), c1(y), c0(z)] * (w0(x) * w1(y) * w0(z))
            + image[batch, channel, c1(x), c1(y), c0(z)] * (w1(x) * w1(y) * w0(z))
            + image[batch, channel, c0(x), c0(y), c1(z)] * (w0(x) * w0(y) * w1(z))
            + image[batch, channel, c1(x), c0(y), c1(z)] * (w1(x) * w0(y) * w1(z))
            + image[batch, channel, c0(x), c1(y), c1(z)] * (w0(x) * w1(y) * w1(z))
            + image[batch, channel, c1(x), c1(y), c1(z)] * (w1(x) * w1(y) * w1(z))
        )

    # Mask transformed image
    image_transformed *= mask

    # Reshape transformed image
    image_transformed = image_transformed.reshape(shape_output)

    # Return results
    if type_output == "positional":
        return image_transformed
    else:
        return {"image": image_transformed}


@overload
def generate_affine(
    parameter_translation: torch.Tensor,
    parameter_rotation: torch.Tensor,
    parameter_scaling: torch.Tensor,
    type_output: Literal["positional"] = "positional",
    type_rotation: ROT_TYPES = "euler_xyz",
    transform_order: ORDER_TYPES = "trs",
) -> torch.Tensor:
    ...


@overload
def generate_affine(
    parameter_translation: torch.Tensor,
    parameter_rotation: torch.Tensor,
    parameter_scaling: torch.Tensor,
    type_output: Literal["named"],
    type_rotation: ROT_TYPES = "euler_xyz",
    transform_order: ORDER_TYPES = "trs",
) -> dict[str, torch.Tensor]:
    ...


@typechecked
def generate_affine(
    parameter_translation: torch.Tensor,
    parameter_rotation: torch.Tensor,
    parameter_scaling: torch.Tensor,
    type_output: RETURN_TYPES = "positional",
    type_rotation: ROT_TYPES = "euler_xyz",
    transform_order: ORDER_TYPES = "trs",
) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
    """Generates an affine transform from translation, rotation, and scaling parameters.

    Args:
        parameter_translation: Translation parameters in pixels between -80 and 80. Must be of shape (B, C, parameters)
            or (B, parameters), with parameters=2 or 3 for 2D and 3D images, respectively.
        parameter_rotation: Rotation parameters in radians. Must be of shape (B, C,  parameters) or (B, parameters),
            with parameters=1, 3 or 4, for 2D, 3D Euler angles, and 3D quaternions, respectively.
        parameter_scaling : Scaling parameters. Must be of shape (B, C,  parameters) or (B, parameters),
            with parameters=2 or 3 for 2D and 3D images, respectively.
        type_rotation: Type of rotation parameters: quaternions or Euler angles. For Euler angles, the order of the
            multiplication of the rotations around x, y, and z is represented in the name  (euler_xyz, euler_yzx, etc.),
            defaults to "euler_xyz"
        transform_order : Order of multiplication of translation, rotation, and scaling transforms, defaults to "trs"
        type_output: Determines how the outputs are returned. If set to "positional", it returns positional outputs.
        If set to "named", it returns a dictionary with named outputs. Defaults to 'positional'.

    Returns:
        transform_affine : torch.Tensor of shape (B, C,  4, 4)
        or dictionary: {"transform_affine": transform_affine}

    Example:
        >>> parameter_translation = torch.rand((1, 3))
        >>> parameter_rotation = torch.rand((1, 4))
        >>> parameter_scaling = torch.rand((1, 3))

        >>> transform_affine = generate_affine(parameter_translation, parameter_rotation, parameter_scaling,\
            type_rotation='quaternions')
    """
    # Validate arguments
    if (
        parameter_translation.shape[:-1] != parameter_rotation.shape[:-1]
        or parameter_translation.shape[:-1] != parameter_scaling.shape[:-1]
    ):
        raise ValueError("mismatched shape of parameters")
    if (
        parameter_translation.device != parameter_rotation.device
        or parameter_translation.device != parameter_scaling.device
    ):
        raise ValueError("mismatched devices of parameters")

    # Generate required transforms
    transform_translation = generate_translation(parameter_translation)
    transform_rotation = generate_rotation(parameter_rotation, type_rotation)
    transform_scaling = generate_scaling(parameter_scaling)

    assert isinstance(transform_translation, torch.Tensor)
    assert isinstance(transform_rotation, torch.Tensor)
    assert isinstance(transform_scaling, torch.Tensor)

    # Sort order of operations
    key = (transform_order.index(x) for x in ("t", "r", "s"))
    operations = (transform_translation, transform_rotation, transform_scaling)
    operations_sorted = [x for _, x in sorted(zip(key, operations))]

    # Generate affine transform
    transform_affine = operations_sorted[0] @ operations_sorted[1] @ operations_sorted[2]

    # Return results
    if type_output == "positional":
        return transform_affine
    else:
        return {"transform_affine": transform_affine}


def generate_translation(
    parameter_translation: torch.Tensor,
    type_output: RETURN_TYPES = "positional",
) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
    """Generates a translation transform from translation parameters.

    Args:
        parameter_translation: Translation parameters. Must be of shape (B, C,  parameters)
            or (B, parameters), with parameters=2 or 3 for 2D and 3D images, respectively.
        type_output: Determines how the outputs are returned. If set to "positional",
            it returns positional outputs. If set to "named", it returns a dictionary with named outputs,
            defaults to "positional"

    Raises:
        ValueError: An error occured because dimension of batched translation parameters is >3
        ValueError: An error occured because length of translation parameters is not 2 or 3
        ValueError: An error occured because type_output is not "positional" or "named"

    Returns:
        transform_translation: torch.Tensor of shape (B, C,  4, 4)
        or dictionary: {"transform_affine": transform_affine}

    Example:
        >>> parameter_translation = torch.rand((1, 3))
        >>> transform_translation = generate_translation(parameter_translation)
    """

    # Validate arguments
    if parameter_translation.dim() not in (1, 2, 3):
        raise ValueError(
            f"expected a 1D, 2D, or 3D parameter_translation, got {parameter_translation.dim()!r}D instead"
        )
    if parameter_translation.shape[-1] not in (2, 3):
        raise ValueError("unexpected shape of parameter_translation")

    # Update variables if required
    device = parameter_translation.device

    # Generate scaling transform
    transform_tiling = (*parameter_translation.shape[:-1], 1, 1)
    transform_translation = torch.eye(parameter_translation.shape[-1] + 1, device=device)
    transform_translation = transform_translation.tile(transform_tiling)

    # Populate translation transform
    transform_translation[..., :-1, -1] = parameter_translation

    # Return results
    if type_output == "positional":
        return transform_translation
    else:
        return {"transform_translation": transform_translation}


def generate_scaling(
    parameter_scaling: torch.Tensor,
    type_output: RETURN_TYPES = "positional",
) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
    """Generates a scaling transform from scaling parameters.

    Args:
        parameter_scaling: Scaling parameters. Must be of shape (B, C,  parameters) or (B, parameters),
            with parameters=2 or 3 for 2D and 3D images, respectively.
        type_output: Determines how the outputs are returned. If set to "positional", it returns positional outputs.
            If set to "named", it returns a dictionary with named outputs. Defaults to positional.

    Returns:
        transform_scaling: tensor of shape (B, C,  4, 4)

    Example:
        >>> parameter_scaling = torch.rand((1, 3))
        >>> transform_scaling = generate_scaling(parameter_scaling)
    """
    # Validate arguments
    if parameter_scaling.dim() not in (1, 2, 3):
        raise ValueError(f"expected a 1D, 2D, or 3D parameter_scaling, got {parameter_scaling.dim()!r}D instead")
    if parameter_scaling.shape[-1] not in (2, 3):
        raise ValueError("unexpected shape of parameter_scaling")

    device = parameter_scaling.device

    # Generate scaling transform
    transform_tiling = (*parameter_scaling.shape[:-1], 1, 1)
    transform_scaling = torch.eye(parameter_scaling.shape[-1] + 1, device=device)
    transform_scaling = transform_scaling.tile(transform_tiling)

    # Populate scaling transform
    for i in range(parameter_scaling.shape[-1]):
        transform_scaling[..., i, i] = parameter_scaling[..., i]

    # Return results
    if type_output == "positional":
        return transform_scaling
    else:
        return {"transform_scaling": transform_scaling}


def generate_rotation(
    parameter_rotation: torch.Tensor,
    type_rotation: ROT_TYPES = "euler_xyz",
    type_output: RETURN_TYPES = "positional",
) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
    """Generates a rotation transform from rotation parameters.

    Args:
        parameter_rotation : Rotation parameters. Must be of shape (B, C,  parameters) or (B, parameters),
            with parameters=1, 3 or 4, for 2D, 3D Euler angles, and 3D quaternions, respectively.
        type_rotation: Type of rotation parameters: quaternions or Euler angles. For Euler angles, the order of the
            multiplication of the rotations around x, y, and z is represented in the
            name (euler_xyz, euler_yzx, etc.) This variable with be ignored for 2D rotations.
            defaults to euler_xyz
        type_output : Determines how the outputs are returned. If set to "positional", it returns positional outputs.
            If set to "named", it returns a dictionary with named outputs. Defaults to positional

    Returns:
        transform_rotation: tensor of shape (B, C,  4, 4)

    Example:
        # Rotation transform for quaternions
        >>> parameter_rotation = torch.rand((1, 4))
        >>> transform_rotation = generate_rotation(parameter_rotation, type_rotation="quaternions")

        # Rotation transform for Euler angles
        >>> parameter_rotation = torch.rand((1, 3))
        >>> transform_rotation = generate_rotation(parameter_rotation, type_rotation="euler_xyz")
    """
    # Validate arguments
    if parameter_rotation.dim() not in (1, 2, 3):
        raise ValueError(f"expected a 1D, 2D, or 3D parameter_rotation, got {parameter_rotation.dim()!r}D instead")
    if parameter_rotation.shape[-1] not in (1, 3, 4):
        raise ValueError("unexpected shape of parameter_scaling")

    if parameter_rotation.shape[-1] != 1:
        if type_rotation[:5] == "euler" and parameter_rotation.shape[-1] != 3:
            raise ValueError("mismatch between type_rotation and shape of parameter_rotation")
        if type_rotation == "quaternions" and parameter_rotation.shape[-1] != 4:
            raise ValueError("mismatch between type_rotation and shape of parameter_rotation")

    device = parameter_rotation.device

    # Generate identity transform
    transform_tiling = (*parameter_rotation.shape[:-1], 1, 1)
    transform_identity = torch.eye(4 - (parameter_rotation.shape[-1] == 1), device=device)
    transform_identity = transform_identity.tile(transform_tiling)

    # Generate and populate 2D rotation transform
    if parameter_rotation.shape[-1] == 1:
        transform_rotation = transform_identity
        transform_rotation[..., 0, 0] = torch.cos(parameter_rotation[..., 0])
        transform_rotation[..., 0, 1] = -torch.sin(parameter_rotation[..., 0])
        transform_rotation[..., 1, 0] = torch.sin(parameter_rotation[..., 0])
        transform_rotation[..., 1, 1] = torch.cos(parameter_rotation[..., 0])

    # Generate and populate 3D Euler rotation transform
    elif parameter_rotation.shape[-1] == 3:
        # Define rotation transform indices for rotation around x, y, and z
        index = {
            "x": torch.tensor([[1, 1, 2, 2], [1, 2, 1, 2], [1, -1, 1, 1]]),
            "y": torch.tensor([[0, 0, 2, 2], [0, 2, 0, 2], [1, 1, -1, 1]]),
            "z": torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1], [1, -1, 1, 1]]),
        }

        # Populate rotation transforms
        transform_rotation = transform_identity.clone()
        for i in range(3):
            i0 = index[type_rotation[6 + i]][0]
            i1 = index[type_rotation[6 + i]][1]
            q0 = index[type_rotation[6 + i]][2].to(device)
            angle = parameter_rotation[..., i]
            transform_temp = transform_identity.clone()
            transform_temp[..., i0, i1] = q0 * torch.stack(
                [torch.cos(angle), torch.sin(angle), torch.sin(angle), torch.cos(angle)], dim=-1
            )
            transform_rotation = transform_rotation.matmul(transform_temp)

    # Generate and populate 3D Quaternion rotation transform
    else:
        # Check if quaternions are normalised
        if torch.any(parameter_rotation.norm(dim=-1) < 1e-5):
            raise ValueError(
                f"parameter_rotation of type Quaternion must be normalised. Got parameter_rotation.norm(dim=-1)\
                ={parameter_rotation.norm(dim=-1)} instead",
            )

        # Separate quaternion components for readability
        if parameter_rotation.dim() == 3:
            q0, q1, q2, q3 = parameter_rotation.permute(dims=(2, 0, 1))
        else:
            q0, q1, q2, q3 = parameter_rotation.permute(dims=(1, 0))

        # Generate rotation transform
        transform_rotation = transform_identity
        transform_rotation[..., 0, 0] = 1 - 2 * (q2**2 + q3**2)
        transform_rotation[..., 0, 1] = 2 * (q1 * q2 - q3 * q0)
        transform_rotation[..., 0, 2] = 2 * (q1 * q3 + q2 * q0)
        transform_rotation[..., 1, 0] = 2 * (q1 * q2 + q3 * q0)
        transform_rotation[..., 1, 1] = 1 - 2 * (q1**2 + q3**2)
        transform_rotation[..., 1, 2] = 2 * (q2 * q3 - q1 * q0)
        transform_rotation[..., 2, 0] = 2 * (q1 * q3 - q2 * q0)
        transform_rotation[..., 2, 1] = 2 * (q2 * q3 + q1 * q0)
        transform_rotation[..., 2, 2] = 1 - 2 * (q1**2 + q2**2)

    # Return results
    if type_output == "positional":
        return transform_rotation
    else:
        return {"transform_rotation": transform_rotation}
