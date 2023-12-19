import torch
from pathlib import Path
from typing import Optional
from ..model_paths import BRAIN_EXTRACTION_MODEL_PATH
from ..tedsnet_multi.network.UNet import UNet_MW as UNet


def load_brainextraction_model(model_path: Optional[Path] = None) -> UNet:
    """ Load the brain extraction model for whole brain masking of the fetal brain.
    Args:
        model_path: the path to the model. Defaults to None, then the standard path is loaded.

    Returns:
        model: brain extraction model with trained weights loaded

    Example:
        >>> model = load_brainextraction_model()
    """

    if model_path is None:
        model_path = BRAIN_EXTRACTION_MODEL_PATH

    model = UNet()
    if torch.cuda.is_available():
        model_weights = torch.load(model_path)
    else:
        model_weights = torch.load(model_path, map_location=torch.device('cpu'))

    model.load_state_dict(model_weights)
    model.eval()
    torch.set_grad_enabled(False)

    return model


def extract_brain(aligned_scan: torch.Tensor, segm_model: UNet) -> torch.Tensor:
    """Generate brain mask for a given aligned scan using the provided segmentation model.

    Args:
        aligned_scan: torch tensor containing the aligned image(s) to segment. Should be of size [B, 1, H, W, D].
            Expected input orientation is equal to the output of the align_to_atlas(scale = False) function.
        segm_model: a loaded segmentation model, can be obtained using the load_segmentation_model() function.

    Returns:
        brain_mask: brain mask of the aligned scan of size [B, 1, H, W, D].

    Example:
        >>> scan = torch.rand((160, 160, 160))
        >>> segm_model = load_segmentation_model()
        >>> brain_mask = extract_brain(aligned_scan, segm_model)
    """
    segm_model.to(aligned_scan.device)
    brain_mask = segm_model(aligned_scan)
    brain_mask = torch.where(brain_mask > 0.5, 1, 0)

    return brain_mask


def extract_scan_brain(scan: torch.Tensor) -> torch.Tensor:
    """Generate brain mask for a given scan using the default brain extraction model weights

    Args:
        scan: torch tensor containing the image(s) to segment. Should be of size [B, 1, H, W, D] with
        pixel values between 0 and 255

    Returns:
        brain_mask: brain mask of the scan of size [B, 1, H, W, D].

    Example:
        >>> scan = torch.rand((160, 160, 160))
        >>> brain_mask = extract_scan_brain(scan)
    """
    extraction_model = load_brainextraction_model()
    brain_mask = extract_brain(scan, extraction_model.to(scan.device))

    return brain_mask
