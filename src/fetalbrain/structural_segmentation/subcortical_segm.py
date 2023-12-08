"""
This module contains the main functions for performing subcortical segmentations on unscaled aligned scans.
As post-processing step, only the largest connected component of each class is kept by default. Doing
this post-processing step can be disabled by setting the connected_component argument to False.

A single scan can be segmented using the :func:`segment_scan_subc` function, which is a wrapper that loads
the segmentation model, prepares the scan for segmentation and predicts the segmentation mask.
    >>> aligned_scan = torch.rand((160, 160, 160))
    >>> segm_pred_np, key_maps = segment_scan_subc(aligned_scan, connected_component=True)

As for the alignment, it is recommended to use the functions :func:`load_segmentation_model`,
:func:`prepare_scan_segm`, and :func:`segment_subcortical` directly to have more control over the
workflow. This can for example be used as follows:
    >>> segm_model = load_segmentation_model()
    >>> aligned_scan = torch.rand((160, 160, 160))
    >>> aligned_scan_prep = prepare_scan_segm(aligned_scan)
    >>> segm_pred, key_maps = segment_subcortical(aligned_scan_prep, segm_model)
    >>> segm_pred_cc = keep_largest_compoment(segm_pred.cpu())

The :func:`segment_subcortical` function can also process batches of data (i.e. multiple scans at once),
which can be useful to speed up analysis. More advanced examples can be found in the Example Gallery.

Lastly, the :func:`compute_volume_segm` function can be used to compute the volume of each structure in the
segmentation mask:
[todo; fix this line] volume_dict = compute_volume_segm(segm_pred, key_maps, spacing=(0.6, 0.6, 0.6))

Module functions
----------------
"""

import torch
from pathlib import Path
from typing import Optional, Union
import numpy as np
import SimpleITK as sitk
from .segmentation_model import UNet
from ..model_paths import SEGM_MODEL_PATH
from ..alignment.align import prepare_scan


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


def segment_subcortical(
    aligned_scan: torch.Tensor, segm_model: torch.nn.DataParallel[UNet]
) -> tuple[torch.Tensor, dict[str, int]]:
    """Generate subcortical predictions for a given aligned scan using the provided segmentation model.

    Args:
        aligned_scan: torch tensor containing the aligned image(s) to segment. Should be of size [B, 1, H, W, D].
            Expected input orientation is equal to the output of the align_to_atlas(scale = False) function.
        segm_model: a loaded segmentation model, can be obtained using the load_segmentation_model() function.

    Returns:
        segm_map: multiclass segmentation map of size [B, H, W, D] with values between 0 and 4. The values
            correspond to the following classes: 0 = background, 1 = choroid plexus (ChP), 2 = lateral posterior
            ventricle horn (LPVH), 3 = cavum septum pellucidum et vergae (CSPV), 4 = cerebellum (CB).
        key_maps: dictionary containing the mapping between the class values and the class names.
    Example:
        >>> aligned_scan = torch.rand((160, 160, 160))
        >>> aligned_scan_prep = prepare_scan_segm(aligned_scan)
        >>> segm_model = load_segmentation_model()
        >>> segm_map, key_maps = segment_subcortical(aligned_scan_prep, segm_model)
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

    # define key maps of model output
    key_maps = {"ChP": 1, "LPVH": 2, "CSPV": 3, "CB": 4}

    return segm_map, key_maps


def compute_volume_segm(
    segm_map: np.ndarray, key_maps: dict[str, int], spacing: tuple[float, float, float]
) -> dict[str, list]:
    """Get the volume of each structure in a segmentation map.

    Args:
        segm_map: the segmentation map of size [B, H, W, D] or [H,W,D] with values corresponding
            to the keys/values in key_maps.
        key_maps: dictionary with the classes as keys, and the integer value in segm_map as value: {class: value}
        spacing: the spacing of the image in mm, should be a tuple of size 3: (x, y, z)

    Returns:
        volume_dict: dictionary with all classes as keys, and a list of the volume of each structure in cm^3 as value.

    Example:
        >>> segm_map = np.random.randint(0, 5, (1, 160, 160, 160))
        >>> key_maps = {"ChP": 1, "LPVH": 2, "CSPV": 3, "CB": 4}
        >>> spacing = (0.6, 0.6, 0.6)
        >>> volume_dict = compute_volume_segm(segm_map, key_maps, spacing)
    """

    if len(segm_map.shape) == 3:
        segm_map = np.expand_dims(segm_map, axis=0)

    # create output list
    volume_dict: dict[str, list] = {class_name: [] for class_name in key_maps.keys()}
    for segm_image in segm_map:
        # compute volume of single voxel (in cm^3)
        voxel_volume_cm = spacing[0] * spacing[1] * spacing[2] / 1000

        # compute volume of each structure in key_maps
        for key in key_maps.keys():
            pixel_count = np.sum(segm_image == key_maps[key])
            volume_cm = pixel_count * voxel_volume_cm
            volume_dict[key].append(volume_cm.item())

    return volume_dict


def keep_largest_compoment(segm_map: np.ndarray) -> np.ndarray:
    """Only keeps the largest connected component for each class

    Args:
        segm_map: input segmentation map of size [B, H, W, D] or [H, W, D] with integers values
            corresponding to a class

    Returns:
        conn_comp_segm: segmentation map of size [B, H, W, D] or [H, W, D] with for each class
            only the largest connected component is kept. The batch dimension is included only if it
            is larger than 1.

    Example:
        >>> segm_map = np.random.randint(0, 5, (1, 160, 160, 160))
        >>> conn_comp_segm = keep_largest_compoment(segm_map)
    """
    input_dim = len(segm_map.shape)
    if input_dim == 3:
        segm_map = np.expand_dims(segm_map, axis=0)

    # create output array
    conn_comp_segm = np.zeros_like(segm_map)

    for im in range(segm_map.shape[0]):
        for classx in np.unique(segm_map)[1:]:
            single_class = np.where(segm_map[im] == classx, 1, 0)

            sitk_im = sitk.GetImageFromArray(single_class)
            # assigns a unique label to each connected component
            labels_cc = sitk.ConnectedComponent(sitk_im)

            # relabels the components so that the largest component is 1
            labels_ordered = sitk.RelabelComponent(labels_cc)

            # get the largest connected component
            largest_cc = labels_ordered == 1

            # convert to numpy array
            largest_cc = sitk.GetArrayFromImage(largest_cc)

            # get the original class_value back and add to array
            largest_cc = largest_cc * classx
            conn_comp_segm[im] += largest_cc

    # remove batch dimension if it was not in input
    if input_dim == 3:
        conn_comp_segm = np.squeeze(conn_comp_segm)

    return conn_comp_segm


def segment_scan_subc(
    aligned_scan: Union[torch.Tensor, np.ndarray], connected_component: bool = True
) -> tuple[np.ndarray, dict]:
    """full function to segment a single scan or a batch of scans

    Args:
        aligned_scan: input array or tensor to segment, can be of size [B,H,W,D], [B,C,H,W,D] or [H,W,D]
        connected_component: whether to only keep the largest connected component. Defaults to True.

    Returns:
        segm_pred_np: segmentation map of size [B, H, W, D] or [H, W, D] with a multi-class segmentation.
            The batch dimension is only kepf it is larger than 1.
        key_maps: dictionary with the classes as keys, and the integer value in segm_map as value:
            {class: value}

    Example:
        >>> aligned_scan = torch.rand((160, 160, 160))
        >>> segm_pred_np, key_maps = segment_scan_subc(aligned_scan)

    """
    segm_model = load_segmentation_model()
    aligned_scan_prep = prepare_scan(aligned_scan)

    segm_pred, key_maps = segment_subcortical(aligned_scan_prep, segm_model)
    segm_pred_np = segm_pred.cpu().numpy()

    if connected_component:
        segm_pred_np = keep_largest_compoment(segm_pred_np)

    return segm_pred_np.squeeze(), key_maps


# to do: write tests for connected components and compute volume segm
