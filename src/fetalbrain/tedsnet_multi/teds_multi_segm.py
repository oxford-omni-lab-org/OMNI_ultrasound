""" How to apply our segmentation method

"""
import torch
import numpy as np
from typing import Optional, Literal
from pathlib import Path
from fetalbrain.tedsnet_multi.network.TEDS_Net import TEDS_Net
from fetalbrain.alignment.align import prepare_scan
from fetalbrain.utils import read_image
from fetalbrain.tedsnet_multi.hemisphere_detector import load_sidedetector_model, detect_side
from ..model_paths import TEDS_MULTI_MODEL_PATH, PRIOR_SHAPE_PATH


def load_tedsmulti_model(model_path: Optional[Path] = None) -> TEDS_Net:
    """Load the trained multistructure segmentation model

    Args:
        model_path: path to the trained model weights

    Returns:
        model: segmentation model with trained weights loaded

    Example:
        >>> model = load_tedsmulti_model()
    """
    if model_path is None:
        model_path = TEDS_MULTI_MODEL_PATH

    model = TEDS_Net()

    model_weights = torch.load(model_path)
    model.load_state_dict(model_weights)

    model.eval()
    torch.set_grad_enabled(False)

    return model


def get_prior_shape_sa(sd: Literal["l", "r"]) -> torch.Tensor:
    """Get the prior paired with each week and side

    Args:
        sd: which side to get the prior shape for, either 'r' or 'l'

    Returns:
        prior_shape: tensor containing the prior shape

    Example:
        >>> prior_shape = get_prior_shape_sa('l')
    """

    # Load in shape prior
    pshape, _ = read_image(PRIOR_SHAPE_PATH)

    # correct for permuted orientation
    pshape_per = np.swapaxes(pshape, 0, 2).astype(int)

    # set the invisible hemisphere to zero, except for the cavum (because it is around the midplane)
    cavum = np.where(pshape_per == 2, 1, 0)
    if sd == "l":
        pshape_per[:, 80:160, :] = 0

    elif sd == "r":
        pshape_per[:, 0:80, :] = 0

    pshape_per = np.where(cavum == 1, 2, pshape_per)

    # One hot the labels
    nclass = 10
    one_hot = np.zeros((nclass, pshape_per.shape[0], pshape_per.shape[1], pshape_per.shape[2]))
    for i in range(1, nclass + 1):
        one_hot[i - 1, :, :, :][pshape_per == i] = 1

    return torch.from_numpy(one_hot.astype(np.float32))


def generate_multiclass_prediction(prediction: torch.Tensor) -> np.ndarray:
    """Convert the TEDS-multiclass output into a multiclass segmentation mask
        Note: I don't think this works for batches atm
    Args:
        prediction: prediction from TEDS model of size [B, 10, H, W, D]

    Returns:
        combined_pred: multiclass segmentation mask of size [H, W, D]
    """
    # due to tedsnets design we approach is as binary for each channel and threshold at 0.4
    pred = (prediction > 0.4).int().squeeze().cpu().numpy()

    # we then combine the predictions into a single multiclass image
    combined_pred = np.zeros_like(pred[0])
    # loop through the channels
    for i, ch in enumerate(range(np.shape(pred)[0])):
        combined_pred = np.where(pred[ch] == 1, i + 1, combined_pred)

    return combined_pred


def segment_tedsall(
    aligned_scan: torch.Tensor, segm_model: TEDS_Net, side: Literal["r", "l"] = "r"
) -> tuple[np.ndarray, dict]:
    """_summary_

    Args:
        aligned_scan: _description_
        segm_model: _description_
        side: _description_. Defaults to "r".

    Returns:
        _description_
    """
    aligned_scan_per = aligned_scan.permute(0, 1, 4, 3, 2)

    # get the prior shape
    prior = torch.unsqueeze(get_prior_shape_sa(side), 0).to(aligned_scan_per.device)

    # forward pass
    logits, _ = segm_model(aligned_scan_per, prior)

    # convert to multiclass [B, H, W, D]
    multiclass = generate_multiclass_prediction(logits.permute(0, 1, 4, 3, 2))

    # define key maps of model output
    key_maps = {
        "Cortical Plate": 1,
        "Cavum Septum": 2,
        "Cerebellum": 3,
        "Choriod Plex": 4,
        "Ventricle": 5,
        "DGM": 6,
        "Thalamus": 7,
        "Brainstem": 8,
        "WM": 9,
        "Frontal Horns": 10,
    }

    return multiclass, key_maps


def segment_scan_tedsall(aligned_scan: torch.Tensor) -> tuple[np.ndarray, dict]:
    """Executes the whole TEDSall segmentation pipeline

    Args:
        aligned_scan: _description_

    Returns:
        _description_
    """

    segm_model = load_tedsmulti_model().to(aligned_scan.device)
    aligned_scan = prepare_scan(aligned_scan)

    # side is now hardcode, this has to change into model prediction
    side_model = load_sidedetector_model()
    side, _ = detect_side(aligned_scan, side_model)

    if side.item() == 0:
        side = "r"
    else:
        side = "l"

    multiclass, keys = segment_tedsall(aligned_scan, segm_model, side="r")
    return multiclass.squeeze(), keys
