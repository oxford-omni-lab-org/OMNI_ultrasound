import torch
from pathlib import Path
from typing import Optional
import numpy as np

# from fetalbrain.alignment.align import _get_transform_to_atlasspace


SIDE_DETECTOR_MODEL_PATH = Path("src/fetalbrain/tedsnet_multi/network/FinalModel_sidedetection.pt")


def load_sidedetector_model(model_path: Optional[Path] = None) -> torch.nn.Module:
    """Load the trained side detection model

    Args:
        model_path: path to the trained model weights. Defaults to None.

    Returns:
        ResNet model with trained weights loaded

    Example:
        >>> model = load_sidedetector_model()
    """
    if model_path is None:
        model_path = SIDE_DETECTOR_MODEL_PATH

    model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=False, num_classes=2)

    model_weights = torch.load(model_path)
    model.load_state_dict(model_weights)

    model.eval()
    torch.set_grad_enabled(False)

    return model


def detect_side(aligned_scan: torch.Tensor, model: torch.nn.Module, from_atlas: bool = True) -> int:
    """_summary_

    Takes as input a scan aligned (no scaling) to the atlas or bean coordinate system.


    Args:
        aligned_scan: _description_
        model: _description_

    Returns:
        pred:

    """
    if torch.max(aligned_scan) <= 1:
        aligned_scan *= 255

    # this is the orientation it was trained on by Maddy
    if not from_atlas:
        midslice = aligned_scan[:, 0, :, :, 79:82]
    # this is equivalent to this in the atlas orientation
    else:
        rotated = torch.permute(aligned_scan, (0, 1, 3, 2, 4))
        midslice = rotated[:, 0, :, :, 85:88]

    # bring channel dimension forward
    midslice = midslice.permute(0, 3, 1, 2)

    outputs = torch.sigmoid(model(midslice)).detach().cpu().numpy()
    pred = np.argmax(outputs)

    return pred, outputs[:, pred]
