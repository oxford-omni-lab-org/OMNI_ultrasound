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
from ..model_paths import TEDS_MULTI_MODEL_PATH, PRIOR_SHAPE_PATH,PRIOR_PARC_PATH


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

    if torch.cuda.is_available():
        model_weights = torch.load(model_path)
    else:
        model_weights = torch.load(model_path, map_location=torch.device('cpu'))

    model.load_state_dict(model_weights)

    model.eval()
    torch.set_grad_enabled(False)

    return model


def get_prior_shape_sa(sd: Literal[0, 1],prior="struc") -> torch.Tensor:
    """Get the prior paired with each week and side

    Args:
        sd: which side to get the prior shape for, either 0 or 1
        prior: which prior shape we are deforming, default "struc"

    Returns:
        prior_shape: tensor containing the prior shape

    Example:
        >>> prior_shape = get_prior_shape_sa(0)

        
    Note: Added prior keyword to allow us to perform parcellation too!
    """

    assert sd in [0, 1], "sd should be either 0 or 1"

    if prior=="struc":
        # Load in shape prior
        pshape, _ = read_image(PRIOR_SHAPE_PATH)
        nclass =10
    else:
        pshape, _ = read_image(PRIOR_PARC_PATH)
        nclass=5

    # correct for permuted orientation
    pshape_per = np.swapaxes(pshape, 0, 2).astype(int)

    if prior=="struc":
        # set the invisible hemisphere to zero, except for the cavum (because it is around the midplane)
        cavum = np.where(pshape_per == 2, 1, 0)
        if sd == 0:
            pshape_per[:, 80:160, :] = 0
        elif sd == 1:
            pshape_per[:, 0:80, :] = 0
        pshape_per = np.where(cavum == 1, 2, pshape_per)

    # One hot the labels
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
    aligned_scan: torch.Tensor, segm_model: TEDS_Net, side: Literal[0, 1] = 0,prior="struc"
) -> tuple[np.ndarray, dict]:
    """_summary_

    Args:
        aligned_scan: _description_
        segm_model: _description_
        side: _description_. Defaults to "0".

    Returns:
        _description_
    """
    aligned_scan_per = aligned_scan.permute(0, 1, 4, 3, 2)

    # get the prior shape
    prior = torch.unsqueeze(get_prior_shape_sa(side,prior), 0).to(aligned_scan_per.device)

    # forward pass
    logits, field = segm_model(aligned_scan_per, prior)

    # convert to multiclass [B, H, W, D]
    multiclass = generate_multiclass_prediction(logits.permute(0, 1, 4, 3, 2))

    # define key maps of model output
    if prior =="struc":
        key_maps = {
            "CoP": 1,
            "CSP": 2,
            "CB": 3,
            "ChP": 4,
            "LV": 5,
            "DGM": 6,
            "Th": 7,
            "BS": 8,
            "WM": 9,
            "FH": 10,
        }
    else:
        key_maps={"FL":1,
                  "PL":2,
                  "TL":3,
                  "OL":4,
                  "IL":5}

    return multiclass, key_maps,field


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

    multiclass, keys,field = segment_tedsall(aligned_scan, segm_model, side=side)
    return multiclass.squeeze(), keys
