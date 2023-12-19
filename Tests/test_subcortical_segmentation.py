import torch
from fetalbrain.structural_segmentation.subcortical_segm import (
    load_segmentation_model,
    segment_subcortical,
    segment_scan_subc,
    keep_largest_compoment,
    compute_volume_segm,
)
from fetalbrain.structural_segmentation.segmentation_model import UNet
from fetalbrain.utils import read_image
from fetalbrain.alignment.align import load_alignment_model, align_to_atlas, prepare_scan, align_scan
from path_literals import TEST_IMAGE_PATH, TEST_SEGM_PATH
import numpy as np


def test_load_segmentation_model() -> None:
    model = load_segmentation_model()
    assert isinstance(model, torch.nn.DataParallel)
    assert isinstance(model.module, UNet)


def test_segment_subcortical() -> None:
    # align scan
    example_scan, _ = read_image(TEST_IMAGE_PATH)
    torch_scan = prepare_scan(example_scan)
    align_model = load_alignment_model()

    # align scan no scaling
    aligned_scan, params = align_to_atlas(torch_scan, align_model, scale=False)

    # prepare scan for segmentation
    segm_model = load_segmentation_model()
    aligned_scan_prep = prepare_scan(aligned_scan)
    multi_class, class_dict = segment_subcortical(aligned_scan_prep, segm_model)

    assert multi_class.shape == (1, 160, 160, 160)
    assert np.max(multi_class) == 4

    # compare to a saved reference segmentation for this volume
    multi_class_np = multi_class[0]

    ref_segmpath = TEST_SEGM_PATH / "ref_segmap.nii.gz"
    ref_segm, _ = read_image(ref_segmpath)

    assert ref_segm.shape == multi_class_np.shape
    assert np.allclose(ref_segm, multi_class_np, atol=1e-4)


def test_segment_scan_subc() -> None:
    example_scan, _ = read_image(TEST_IMAGE_PATH)
    aligned_scan, _ = align_scan(example_scan)

    segm, keys = segment_scan_subc(aligned_scan, connected_component=False)
    assert segm.shape == (160, 160, 160)

    ref_segmpath = TEST_SEGM_PATH / "ref_segmap.nii.gz"
    ref_segm, _ = read_image(ref_segmpath)

    assert ref_segm.shape == segm.shape
    assert np.allclose(ref_segm, segm, atol=1e-4)


def test_keep_largest_component() -> None:
    # get a segmentation
    ref_segmpath = TEST_SEGM_PATH / "ref_segmap.nii.gz"
    ref_segm, _ = read_image(ref_segmpath)

    # check that it doesn't do anything for segm without unconnected
    assert np.all(ref_segm == keep_largest_compoment(ref_segm))

    # add a unconnected component to the label map
    segm_cc = ref_segm.copy()
    segm_cc[0:2, :2, :2] = 1
    assert not np.allclose(segm_cc, ref_segm, atol=1e-4)

    # prcoess this to remove the component
    segm_cc_rem = keep_largest_compoment(segm_cc)

    # check that the unconnected component was removed
    assert np.all(segm_cc_rem == ref_segm)


def test_compute_volume_segm() -> None:
    ref_segmpath = TEST_SEGM_PATH / "ref_segmap.nii.gz"
    ref_segm, spacing = read_image(ref_segmpath)

    # compute volume of each class
    key_maps = {"ChP": 1, "LPVH": 2, "CSPV": 3, "CB": 4}
    volume_dict = compute_volume_segm(ref_segm, key_maps=key_maps, spacing=spacing)

    assert volume_dict.keys() == key_maps.keys()
    assert np.isclose(volume_dict["ChP"], 0.692, atol=1e-3)
    assert np.isclose(volume_dict["LPVH"], 0.179, atol=1e-3)
    assert np.isclose(volume_dict["CSPV"], 0.142, atol=1e-3)
    assert np.isclose(volume_dict["CB"], 1.696, atol=1e-3)
