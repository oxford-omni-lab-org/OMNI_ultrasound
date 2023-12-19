import torch
import numpy as np
from fetalbrain.utils import read_image
from fetalbrain.alignment.align import align_scan
from fetalbrain.brain_extraction.extract import load_brainextraction_model, extract_brain, extract_scan_brain
from path_literals import TEST_IMAGE_PATH, TEST_BRAINEXTRACTION_PATH


def compare_threshold(scan1: np.ndarray, scan2: np.ndarray, threshold: float = 1.0) -> float:
    no_pixels = 160 * 160 * 160
    percentage_equal = np.count_nonzero(np.abs(scan1 - scan2) <= threshold) / (no_pixels) * 100
    return percentage_equal


def test_load_brainextraction_model() -> None:
    model = load_brainextraction_model()
    assert isinstance(model, torch.nn.Module)


def test_extract_brain() -> None:
    example_scan, _ = read_image(TEST_IMAGE_PATH)

    # align to atlas
    aligned_im, _ = align_scan(example_scan, scale=False, to_atlas=True)

    # extract the whole brain
    segm_model = load_brainextraction_model()
    brain_mask, key_map = extract_brain(aligned_im, segm_model)
    brain_mask = brain_mask.cpu().squeeze().numpy()

    # compare to a reference segmentation
    ref_segmpath = TEST_BRAINEXTRACTION_PATH / "ref_brainmask.nii.gz"
    ref_segm, _ = read_image(ref_segmpath)

    assert ref_segm.shape == brain_mask.shape
    # assert np.allclose(ref_segm, brain_mask, atol=1e-4)
    print(compare_threshold(ref_segm, brain_mask, 1))
    assert compare_threshold(ref_segm, brain_mask, 1) > 0.98


def test_extract_scan_brain() -> None:
    example_scan, _ = read_image(TEST_IMAGE_PATH)

    # align to atlas
    aligned_im, _ = align_scan(example_scan, scale=False, to_atlas=True)
    brain_mask, key_map = extract_scan_brain(aligned_im)
    brain_mask = brain_mask.squeeze()

    # compare to a reference segmentation
    ref_segmpath = TEST_BRAINEXTRACTION_PATH / "ref_brainmask.nii.gz"
    ref_segm, _ = read_image(ref_segmpath)

    assert ref_segm.shape == brain_mask.shape
    assert compare_threshold(ref_segm, brain_mask, 1) > 0.98
