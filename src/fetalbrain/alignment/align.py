"""This module contains the main functions for aligning the scans.

A single scan be aligned using the :func:`align_scan` function, which is a wrapper that loads the alignment model,
prepares the scan into pytorch and computes and applies the alignment transformation. The alignment can be
applied without scaling (i.e. preserving the size of the brain) or with scaling (i.e. scaling all images to the
same reference brain size at 30GWs):
    >>> dummy_scan = np.random.rand(160, 160, 160)
    >>> aligned_scan, params = align_scan(dummy_scan, scale=False, to_atlas=True)
    >>> aligned_scan_scaled, params = align_scan(dummy_scan, scale=True)

For aligning a large number of scans,
it is recommended to access the functions :func:`load_alignment_model`, :func:`prepare_scan` and the
:func:`align_to_atlas` functions directly so that the alignment model is not reloaded for the alignment of each scan.
For example as follows:
    >>> model = load_alignment_model()
    >>> dummy_scan = np.random.rand(160, 160, 160)
    >>> torch_scan = prepare_scan(dummy_scan)
    >>> aligned_scan, params = align_to_atlas(torch_scan, model, scale = False)

The :func:`align_to_atlas` function can also process batches of data (i.e. multiple scans at once), which can be useful
to speed up analysis. More advanced examples can be found in the Example Gallery.

Module functions
-----------------
"""
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Union, overload, Literal
from typeguard import typechecked
from .kelluwen_transforms import generate_affine, apply_affine
from .fBAN_v1 import AlignmentModel
from ..model_paths import BAN_MODEL_PATH


BEAN_TO_ATLAS = Path("src/fetalbrain/alignment/config/25wks_Atlas(separateHems)_mean_warped.json")


def load_alignment_model(model_path: Optional[Path] = None) -> AlignmentModel:
    """Load the fBAN alignment model
    Args:
        model_path: path to the trained model, defaults to None (uses the default model)

    Returns:
        model: alignment model with trained weights loaded

    Example:
        >>> model = load_alignment_model()
    """

    if model_path is None:
        model_path = BAN_MODEL_PATH

    # instantiate model
    model = AlignmentModel()
    model_weights = torch.load(model_path)
    model.load_state_dict(model_weights, strict=True)

    # set to eval mode
    model.eval()
    torch.set_grad_enabled(False)

    return model


@typechecked
def prepare_scan(image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """prepares the scan for subcortical segmentation

    Args:
        image: numpy array or tensor of size
            [B, C, H, W, D], or [B, H, W, D], or [H, W, D]

    Returns:
        :tensor of size [B, C, H, W, D] with values between 0 and 255

    Example:
        >>> image = np.random.random_sample((1, 1, 160, 160, 160))
        >>> image = prepare_scan_segm(image)
        >>> assert torch.max(image) > 1

        >>> image = torch.rand((1, 1, 160, 160, 160))
        >>> image = prepare_scan_segm(image)
        >>> assert torch.max(image) > 1
    """

    # convert to torch tensor
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).float()

    # add channel and batch dimensions
    if len(image.shape) == 3:
        image = image.unsqueeze(0).unsqueeze(0)
    elif len(image.shape) == 4:
        image = image.unsqueeze(1)

    # scale between 0 and 1
    if torch.max(image) <= 1:
        image = image * 255

    return image


# Function overload to make mypy recognize different return types
@overload
def align_to_bean(
    image: torch.Tensor, model: AlignmentModel, return_affine: Literal[False] = False, scale: bool = False
) -> tuple[torch.Tensor, dict]:
    ...


@overload
def align_to_bean(
    image: torch.Tensor, model: AlignmentModel, return_affine: Literal[True], scale: bool = False
) -> tuple[torch.Tensor, dict, torch.Tensor]:
    ...


def align_to_bean(
    image: torch.Tensor, model: AlignmentModel, return_affine: bool = False, scale: bool = False
) -> Union[tuple[torch.Tensor, dict], tuple[torch.Tensor, dict, torch.Tensor]]:
    """Aligns the scan to the bean coordinate system using the fban model.

    Args:
        image:  tensor of size [B,1,H,W,D] containing the image(s) to align between 0 and 255
        model: the model used for inference
        return_affine: whether to return the affine transformation, defaults to False
        scale: whether to apply scaling, defaults to False

    Returns:
        aligned_image: tensor of size [B,1,H,W,D] containing the aligned image(s)
        params: dictionary containing the applied parameters
        affine (optional): tensor containing the affine transformation of size [B,4,4]

    Example:
        >>> model = load_alignment_model()
        >>> dummy_scan = np.random.rand(160, 160, 160)
        >>> torch_scan = prepare_scan(dummy_scan)
        >>> aligned_scan, params = align_to_bean(torch_scan, model)
    """

    model.to(image.device)
    translation, rotation, scaling = model(image)

    if scale is False:
        scaling = torch.tensor([[1.0, 1.0, 1.0]], device=image.device)

    # generate affine transform
    transform_affine = generate_affine(
        parameter_translation=translation * 160,
        parameter_rotation=rotation,
        parameter_scaling=scaling,
        type_rotation="quaternions",
        transform_order="srt",
    )

    # apply transform to image
    assert type(transform_affine) is torch.Tensor
    image_aligned = apply_affine(image, transform_affine)
    assert type(image_aligned) is torch.Tensor

    # interpolation sometimes introduces values outside range
    image_aligned = torch.clamp(image_aligned, 0, 255)

    # make dict from parameters
    param_dict = {"translation": translation, "rotation": rotation, "scaling": scaling}

    if return_affine:
        return image_aligned, param_dict, transform_affine
    else:
        return image_aligned, param_dict


def align_to_atlas(
    image: torch.Tensor, model: AlignmentModel, scale: bool = False, return_affine: bool = False
) -> Union[tuple[torch.Tensor, dict], tuple[torch.Tensor, dict, torch.Tensor]]:
    """
    Aligns the scan to the atlas coordinate system using the fban model.
    The function makes a prediction to go from the orientation to the bean coordinates, and then applies
    an additional transformation to go to the atlas orientation. The bean to atlas transformation is only well defined
    for scaled image volumes, so the affine transformation is always generated including scaling.
    If scaling is set to False, the inverse scaling transform is applied after transformation to the atlas space.

    data flow (scale = True): unscaled bean --> unscaled atlas space -> scaled atlas space \n
    data flow (scale = False): unscaled bean --> unscaled atlas space

    Args:
        image: tensor of size [B,1,H,W,D] containing the image(s) to align with pixel values between 0 and 255
        model: the model used for inference
        scale: whether to apply scaling, defaults to False
        return_affine: whether to return the affine transformation, defaults to False

    Returns:
        aligned_to_atlas_scan: tensor of size [B,1,H,W,D] containing the aligned image(s)
        param_dict: dictionary containing the applied parameters
        affine (optional): tensor containing the affine transformation of size [B,4,4]

    Example:
        >>> model = load_alignment_model()
        >>> dummy_scan = np.random.rand(160, 160, 160)
        >>> torch_scan = prepare_scan(dummy_scan)
        >>> aligned_scan, params = align_to_atlas(torch_scan, model)
    """

    model.to(image.device)

    translation, rotation, scaling = model(image)

    # generate affine transform without scaling
    transform_affine_noscale = generate_affine(
        parameter_translation=translation * 160,
        parameter_rotation=rotation,
        parameter_scaling=torch.tensor([[1.0, 1.0, 1.0]]),
        type_rotation="quaternions",
        transform_order="srt",
    )

    # get bean to atlas transformation
    to_atlas_affine = _get_transform_to_atlasspace()

    if not scale:
        total_transform = to_atlas_affine @ transform_affine_noscale
    else:
        # generate scaling transform to apply after
        scaling_affine = scaling[0, 0] * torch.eye(4, 4).to(image.device)
        scaling_affine[3, 3] = 1

        # alignment -> to_atlas -> scaling
        total_transform = scaling_affine @ to_atlas_affine @ transform_affine_noscale

    # apply whole transformation in one
    aligned_to_atlas_scan = apply_affine(image, total_transform)
    assert isinstance(aligned_to_atlas_scan, torch.Tensor)

    # interpolation sometimes introduces values outside range
    aligned_to_atlas_scan = torch.clamp(aligned_to_atlas_scan, 0, 255)

    param_dict = {"translation": translation, "rotation": rotation, "scaling": scaling}

    if return_affine:
        return aligned_to_atlas_scan, param_dict, total_transform
    else:
        return aligned_to_atlas_scan, param_dict


def transform_from_params(
    image: torch.Tensor,
    translation: Optional[torch.Tensor] = None,
    rotation: Optional[torch.Tensor] = None,
    scaling: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Transforms the images in the input batch with the given translation, rotation and scaling parameters.
    If no parameters are given for a certain transformation, the default values that have no effect are used.

    Args:
        image: tensor of size [B,1, H,W,D] containing the image(s) to align
        translation: tensor with size [B,3] containing translation for each axis between 0 and 1, defaults to None
        rotation: tensor with size [B,4] containing rotation quarternions, defaults to None
        scaling: tensor with size [B,3] containing the scaling parameters, defaults to None

    Returns:
        image_aligned: tensor of size [B,1,H,W,D] containing the aligned image(s)

    Example:
        >>> dummy_scan = np.random.rand(160, 160, 160)
        >>> torch_scan = prepare_scan(dummy_scan)
        >>> translation = torch.tensor([[0.1, 0.05, 0.1]])
        >>> aligned_scan = transform_from_params(torch_scan, translation=translation)
    """
    # set all parameters to default values if they are not given
    if translation is None:
        translation = torch.tensor([[0.0, 0.0, 0.0]])
    if scaling is None:
        scaling = torch.tensor([[1.0, 1.0, 1.0]])
    if rotation is None:
        rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    # generate affine transform
    device = image.device
    transform_affine = generate_affine(
        parameter_translation=translation.to(device) * 160,
        parameter_rotation=rotation.to(device),
        parameter_scaling=scaling.to(device),
        type_rotation="quaternions",
        transform_order="srt",
    )

    # apply transform to image
    assert type(transform_affine) is torch.Tensor
    image_aligned = apply_affine(image, transform_affine)
    assert type(image_aligned) is torch.Tensor

    return image_aligned


@typechecked
def transform_from_affine(image: torch.Tensor, transform_affine: torch.Tensor) -> torch.Tensor:
    """Applies the given affine transformation to the input batch of images

    Args:
        image: tensor of size [B,1,H,W,D] containing the image(s) to align
        transform_affine: tensor of size [B,4,4] containing the affine transformation(s)

    Returns:
        image_transformed: tensor containing the aligned image

    Example:
        >>> dummy_scan = np.random.rand(160, 160, 160)
        >>> torch_scan = prepare_scan(dummy_scan)
        >>> transform_identity = torch.eye(4,4).unsqueeze(0)  # identity transform
        >>> aligned_scan = transform_from_affine(torch_scan, transform_identity)
    """

    # apply transform to image
    image_transformed = apply_affine(image, transform_affine, type_resampling="bilinear", type_origin="centre")
    assert type(image_transformed) is torch.Tensor

    return image_transformed


def _get_atlastransform_itksnap(pixel_spacing: float = 0.6) -> torch.Tensor:
    """These transformation parameters were obtained by registering a set of 20 bean aligned images (no scale)
        to the same images aligned to the atlas space in ITK-SNAP. There was no clear pattern in the
        parameters with respect to GA/skull size, so a median was taken which can be used to align the
        unscaled BEAN images to the atlas space.

        In ITK-SNAP the parameters are given in degree and mm, so these have to be transformed to
        pixels and radians to be used with the transformation functions.

    Args:
        pixel_spacing: pixel spacing of the images, defaults to 0.6

    Returns:
        atlas_transform: tensor of size (1, 4, 4)

    Example:
        >>> atlas_transform = _get_atlastransform_itksnap()
        >>> assert atlas_transform.shape == (1,4,4)
    """
    eu_params_itk = [-1.75, 6.78, 0.69]  # in degree
    tr_params_itk = [-5.62, -0.306, -2.451]  # in mm

    # convert the parameters to radians and mm
    eu_params = -torch.Tensor(eu_params_itk).reshape(1, -1) * np.pi / 180  # in rad
    tr_params = -torch.Tensor(tr_params_itk).reshape(1, -1) / pixel_spacing  # in mm

    # the scaling parameters (unchanged)
    sc_param_to_atlas = torch.Tensor([1, 1, 1]).reshape(1, -1)

    # the type_rotation and transform_order has been set to exactly match ITK-SNAP output
    atlas_transform = generate_affine(
        tr_params, eu_params, sc_param_to_atlas, type_rotation="euler_xyz", transform_order="rts"
    )
    return atlas_transform


def _get_transform_to_atlasspace() -> torch.Tensor:
    """This function computes the total transform to go from BEAN aligned to the atlas space.
    It first permutes the axis to match the planes of the atlas space, and then applies a
    small affine transformation that translates + rotates the images to the atlas space.

    Returns:
        total_transform: transformation matrix of size [1,4,4]

    :example:
        >>> atlas_transform = _get_transform_to_atlasspace()
        >>> assert atlas_transform.shape == (1,4,4)
    """

    atlas_transform = _get_atlastransform_itksnap()

    # these are permutations so that the transformation matrices align with the images
    first_perm = torch.tensor([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32).unsqueeze(
        0
    )

    # obtain the complete transformation
    total_transform = atlas_transform @ first_perm

    return total_transform


def align_scan(scan: np.ndarray, scale: bool = False, to_atlas: bool = True) -> tuple[np.ndarray, dict]:
    """align a scan to a reference coordinate system

    This function aligns the input scan to either the atlas or bean coordinate system, with the
    atlas space as default. This function is a wrapper that loads the alignment model, prepares
    the scan and computes and applies the transformation.

    Args:
        scan: array containing the scan of size [H,W,D]
        scale: whether to apply scaling. Defaults to False.
        to_atlas: whether to align to the atlas coordinate system, otherwise
            the BEAN coordinate system is used (internal use). Defaults to True.

    Returns:
        aligned_scan: the aligned scan

    Example:
        >>> dummy_scan = np.random.rand(160, 160, 160)
        >>> aligned_scan, params = align_scan(dummy_scan)
    """

    # load model
    model = load_alignment_model()

    # prepare scan
    torch_scan = prepare_scan(scan)

    # align scan
    if to_atlas:
        aligned_scan, params = align_to_atlas(torch_scan, model, scale=scale)  # type: ignore
    else:
        aligned_scan, params = align_to_bean(torch_scan, model, scale=scale)

    return aligned_scan, params  # type: ignore
