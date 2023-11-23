"""
This module contains the main access functions for aligning the input scan to a reference coordinate system.
"""
import json
import torch
import numpy as np
from pathlib import Path
import sys
from typing import Optional, Union, overload, Literal
from typeguard import typechecked

sys.path.append("/home/sedm6226/Documents/Projects/US_analysis_package")

from src.alignment.kelluwen_transforms import generate_affine, apply_affine  # noqa: E402
from src.alignment.fBAN_v1 import AlignModel  # noqa: E402

BAN_MODEL_PATH = Path("src/alignment/config/model_weights.pt")
BEAN_TO_ATLAS = Path("src/alignment/config/25wks_Atlas(separateHems)_mean_warped.json")


def load_alignment_model(model_path: Optional[Path] = None) -> AlignModel:
    """Load the fBAN alignment model

    :param model_path: path to the trained model, defaults to None (uses the default model)
    :return: alignment model with trained weights loaded
    """
    if model_path is None:
        model_path = BAN_MODEL_PATH

    # instantiate model
    model = AlignModel()
    model_weights = torch.load(model_path)
    model.load_state_dict(model_weights, strict=True)

    # set to eval mode
    model.eval()
    torch.set_grad_enabled(False)

    return model


@typechecked
def prepare_scan(image: np.ndarray) -> torch.Tensor:
    """Prepares the scan to be aligned with the alignment model.
    Normalises the scan between 0 and 1 and transforms the numpy array to pytorch.

    :param example_scan: scan to be normalized and transformed
    :return: pytorch tensor of the scan with pixel values between 0 and 1
    """

    torch_image = torch.from_numpy(image)

    # normalise if needed
    if np.max(image) > 1:
        torch_image -= torch_image.min()
        torch_image /= torch_image.max()

    torch_image = torch_image[None, None, :]
    torch_image = torch_image.float()

    return torch_image


# Function overload to make mypy recognize different return types
@overload
def align_to_bean(
    image: torch.Tensor, model: AlignModel, return_affine: Literal[False] = False, scale: bool = True
) -> tuple[torch.Tensor, dict]:
    ...


@overload
def align_to_bean(
    image: torch.Tensor, model: AlignModel, return_affine: Literal[True], scale: bool = True
) -> tuple[torch.Tensor, dict, torch.Tensor]:
    ...


def align_to_bean(
    image: torch.Tensor, model: AlignModel, return_affine: bool = False, scale: bool = True
) -> Union[tuple[torch.Tensor, dict], tuple[torch.Tensor, dict, torch.Tensor]]:
    """Aligns the scan to the bean coordinate system using the fban model.

    :param image: tensor of size [B,1,H,W,D] containing the image(s) to align
    :param model: the model used for inference
    :param scale: whether to apply scaling, defaults to True
    :return affine: whether to return the affine transformation, defaults to False
    :return: a tuple of the aligned image [B,1,H,W,D] and a dictionary containing the applied parameters or
            a tuple of the aligned image [B,1,H,W,D], a diciontary containing the applied parameters
            and a tensor containing the  affine transformation of size [B,4,4].
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

    # make dict from parameters
    param_dict = {"translation": translation, "rotation": rotation, "scaling": scaling}

    if return_affine:
        return image_aligned, param_dict, transform_affine
    else:
        return image_aligned, param_dict


def align_to_atlas(
    image: torch.Tensor, model: AlignModel, scale: bool = False, return_affine: bool = False
) -> Union[tuple[torch.Tensor, dict], tuple[torch.Tensor, dict, torch.Tensor]]:
    """
    Aligns the scan to the atlas coordinate system using the fban model.
    The function makes a prediction to go from the orientation to the bean coordinates, and then applies
    an additional transformation to go to the atlas orientation. The bean to atlas transformation is only well defined
    for scaled image volumes, so the affine transformation is always generated including scaling.
    If scaling is set to False, the inverse scaling transform is applied after transformation to the atlas space.

    Flow (scale = True): scaled bean --> scaled atlas space
    Flow (scale = False): scaled bean --> scaled atlas space --> unscaled atlas space

    :param image: tensor of size [B,1,H,W,D] containing the image(s) to align
    :param model: the model used for inference
    :param scale: whether to apply scaling, defaults to True
    :return affine: whether to return the affine transformation, defaults to False
    :return: a tuple of the aligned image [B,1,H,W,D] and a dictionary containing the applied parameters or
            a tuple of the aligned image [B,1,H,W,D], a diciontary containing the applied parameters
            and a tensor containing the  affine transformation of size [B,4,4].
    """

    model.to(image.device)
    translation, rotation, scaling = model(image)

    # generate affine transform with scaling
    transform_affine = generate_affine(
        parameter_translation=translation * 160,
        parameter_rotation=rotation,
        parameter_scaling=scaling,
        type_rotation="quaternions",
        transform_order="srt",
    )

    # get bean to atlas transformation
    to_atlas_affine = _get_transform_to_atlasspace()

    if not scale:
        # construct scaling transform
        scaling_affine = scaling[0, 0] * torch.eye(4, 4).to(image.device)
        scaling_affine[3, 3] = 1

        # scaling + alignment  -> to_atlas -> undo scaling
        total_transform = scaling_affine.inverse() @ to_atlas_affine @ transform_affine

    else:
        # scaling + alignment  -> to_atlas -> undo scaling
        total_transform = to_atlas_affine @ transform_affine

    # apply whole transformation in one
    aligned_to_atlas_scan = apply_affine(image, total_transform)
    assert isinstance(aligned_to_atlas_scan, torch.Tensor)

    param_dict = {"translation": translation, "rotation": rotation, "scaling": scaling}

    if return_affine:
        return aligned_to_atlas_scan, param_dict, transform_affine
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

    :param image: tensor of size [B,1, H,W,D] containing the image(s) to align
    :param translation: tensor with size [B,3] containing translation for each axis, defaults to None
    :param rotation: tensor with size [B,4] containing rotation quarternions, defaults to None
    :param scaling: tensor with size [B,3] containing the scaling parameters, defaults to None
    :return: tensor containing the aligned image
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

    :param image: tensor of size [B,1,H,W,D] containing the image(s) to align
    :param transform_affine: tensor of size [B,4,4] containing the affine transformation(s)
    :return: tensor containing the aligned image
    """

    # apply transform to image
    image_aligned = apply_affine(image, transform_affine, type_resampling="bilinear", type_origin="centre")
    assert type(image_aligned) is torch.Tensor

    return image_aligned


def _get_atlastransform() -> torch.Tensor:
    """This generates the affine transformation matrix to go from the bean space to the atlas space

    :return: _description_
    """

    params_to_atlas = json.load(open(BEAN_TO_ATLAS, "r"))
    eu_params = params_to_atlas["eu_param"]
    tr_params = params_to_atlas["tr_param"]

    eu_param_to_atlas = torch.Tensor(eu_params).reshape(1, -1)
    tr_param_to_atlas = torch.Tensor(tr_params).reshape(1, -1)
    sc_param_to_atlas = torch.Tensor([1, 1, 1]).reshape(1, -1)

    # the negative, likely going from left to right handed coordinate system
    atlas_transform = generate_affine(
        tr_param_to_atlas, -eu_param_to_atlas, sc_param_to_atlas, type_rotation="euler_zyx", transform_order="trs"
    )

    # scale the scaling values to -80 - 80 (instead of -1 - 1 which was used in older versions)
    atlas_transform[0, :3, 3] = atlas_transform[0, :3, 3] * 80

    return atlas_transform


def _get_transform_to_atlasspace() -> torch.Tensor:
    """This incorporates some permutations (implemented as rotation matrices) with the atlas transformation to go
    directly from aligned images in bean orientation to atlas space

    :return: transformation matrix of size [1,4,4]
    """

    atlas_transform = _get_atlastransform()

    # these are permutations so that the transformation matrices align with the images
    first_perm = torch.tensor([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=torch.float32).unsqueeze(
        0
    )
    # the second flips the hemisphere, this one is not strictly required, check for consistency in data
    second_perm = torch.tensor(
        [[0, 1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=torch.float32
    ).unsqueeze(0)

    # obtain the complete transformation
    total_transform = second_perm @ atlas_transform @ first_perm

    return total_transform
