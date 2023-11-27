import torch
from pathlib import Path
from typing import Optional, Union
import sys
import numpy as np
from typeguard import typechecked

sys.path.append("/home/sedm6226/Documents/Projects/US_analysis_package")
from src.structural_segmentation.segmentation_model import UNet  # noqa: E402

SEGM_MODEL_PATH = Path(
    "/mnt/data/Projects_Results/subcortical_segmentation/") / 'Final_Experiments2' / \
    "Results" / "03_11_2021_26_SingleVolumesAligned_20training" / 'run_0' / \
    "modelcheckpoint_epoch_999_loss0.042063.tar"


def load_segmentation_model(model_path: Optional[Path] = None) -> torch.nn.DataParallel[UNet]:
    """Load the trained segmentation model
    Args:
        model_path: path to the trained model. Defaults to None (uses the default model).

    Returns:
        model: segmentation model with trained weights loaded

    Example:
        >>> model = load_segmentation_model()
    """
    if model_path is None:
        model_path = SEGM_MODEL_PATH

    # instantiate model
    # datapar
    model = torch.nn.DataParallel(UNet(1, 5, min_featuremaps=16, depth=5))

    # load model weights
    model_weights = torch.load(model_path)
    model.load_state_dict(model_weights["model_state_dict"])

    # set model to evaluation mode
    model.eval()
    torch.set_grad_enabled(False)

    return model


@typechecked
def prepare_scan_segm(image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """prepares the scan for subcortical segmentation

    Args:
        image: numpy array or tensor of size
            [B, C, H, W, D], or [C, H, W, D], or [H, W, D]

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
        image = image.unsqueeze(0)

    # scale between 0 and 1
    if torch.max(image) <= 1:
        image = image * 255

    return image


def segment_subcortical(aligned_scan: torch.Tensor, segm_model: torch.nn.DataParallel[UNet]) -> torch.Tensor:
    """Generate subcortical predictions for a given aligned scan using the provided segmentation model.

    Args:
        aligned_scan: torch tensor containing the aligned image(s) to segment. Should be of size [B, 1, H, W, D].
            Expected input orientation is equal to the output of the align_to_atlas(scale = False) function.
        segm_model: a loaded segmentation model, can be obtained using the load_segmentation_model() function.

    Returns:
        segm_map: multiclass segmentation map of size [B, H, W, D] with values between 0 and 4. The values
            correspond to the following classes: 0 = background, 1 = choroid plexus (ChP), 2 = lateral posterior
            ventricle horn (LPVH), 3 = cavum septum pellucidum et vergae (CSPV), 4 = cerebellum (CB).

    Example:
        >>> aligned_scan = torch.rand((160, 160, 160))
        >>> aligned_scan_prep = prepare_scan_segm(aligned_scan)
        >>> segm_model = load_segmentation_model()
        >>> segm_map = segment_subcortical(aligned_scan_prep, segm_model)
        >>> assert segm_map.shape == (1, 160, 160, 160)

    """
    # because we load / save now with nii format, we need to permute the dimensions so that it matches the model's
    # expected input
    aligned_scan_per = aligned_scan.permute(0, 1, 4, 3, 2)

    # forward pass
    logits = segm_model(aligned_scan_per)

    # softmax activation
    output = torch.softmax(logits, dim=1).permute(0, 1, 4, 3, 2)

    # get multilabel segmentation map
    segm_map = output.argmax(dim=1)

    return segm_map


# def add largest connected component function (preferably without dependencies)
# def add function to get subcortical volumes