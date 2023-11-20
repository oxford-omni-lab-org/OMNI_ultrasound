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
CONFIG_PATH = Path("src/alignment/config/model_configuration.json")


def load_alignment_model(model_path: Optional[Path] = None) -> AlignModel:
    """Load the fBAN alignment model

    :param model_path: path to the trained model, defaults to None (uses the default model)
    :return: alignment model with trained weights loaded
    """
    if model_path is None:
        model_path = BAN_MODEL_PATH

    # Get config
    config_path = CONFIG_PATH
    model_config = json.load(open(config_path, "r"))

    # instantiate model
    model = AlignModel(**model_config)
    model_weights = torch.load(model_path)
    model.load_state_dict(model_weights, strict=True)

    # set to eval mode
    model.eval()
    torch.set_grad_enabled(False)

    return model


@typechecked
def prepare_scan(image: np.ndarray) -> torch.Tensor:
    """noramlize the scan between 0 and 1 and transform to pytorch

    :param example_scan: scan to be normalized and transformed
    :return: pytorch tensor of the scan between 0 and 1
    """

    torch_image = torch.from_numpy(image)

    # normalise if needed
    if np.max(image) > 1:
        torch_image -= torch_image.min()
        torch_image /= torch_image.max()

    torch_image = torch_image[None, None, :]
    torch_image = torch_image.float()

    return torch_image


# Function overlead to make mypy recognize different return types
@overload
def align_scan(
    image: torch.Tensor, model: AlignModel, return_affine: Literal[False] = False, scale: bool = True
) -> tuple[torch.Tensor, dict]:
    ...


@overload
def align_scan(
    image: torch.Tensor, model: AlignModel, return_affine: Literal[True], scale: bool = True
) -> tuple[torch.Tensor, dict, torch.Tensor]:
    ...


def align_scan(
    image: torch.Tensor,
    model: AlignModel,
    return_affine: bool = False,
    scale: bool = True
) -> Union[tuple[torch.Tensor, dict], tuple[torch.Tensor, dict, torch.Tensor]]:
    """align the scan using the fban model and return the aligned scan and the parameters

    :param image: the image to align [1,1,H,W,D]
    :param model: the model used for inference
    :param scale: whether to apply scaling, defaults to True
    :return affine: whether to return the affine transformation, defaults to False
    :return: a tuple of the aligned image [1,1,H,W,D] and the parameters or
            a tuple of the aligned image [1,1,H,W,D], the parameters and the affine transformation [1,4,4]
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


def align_from_params(
    image: torch.Tensor,
    translation: Optional[torch.Tensor] = None,
    rotation: Optional[torch.Tensor] = None,
    scaling: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """align the scan with the transformation parameters given

    :param image: image to align
    :param translation: size [1,3] containing translation for each axis, defaults to None
    :param rotation: size [1,4] containing rotation quarternions, defaults to None
    :param scaling: size [1,3] containing the scaling parameters, defaults to None
    :return: aligned image
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


def unalign_scan(aligned_image: torch.Tensor, transform_affine: torch.Tensor) -> torch.Tensor:
    """unalign the scan and return the aligned scan and the parameters

    :param aligned_image: image [1, 1, H, W, D]
    :param transform_affine: the alignment transformation [1, 4, 4]
    :return: image back transformed to the original acquisition alignment
    """

    inverse_transform = torch.inverse(transform_affine.squeeze()).unsqueeze(0)
    unaligned_image = apply_affine(aligned_image, inverse_transform)
    return unaligned_image
