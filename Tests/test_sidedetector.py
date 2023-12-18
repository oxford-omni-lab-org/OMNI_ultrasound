import torch
import numpy as np
from fetalbrain.utils import read_image
from fetalbrain.tedsnet_multi.hemisphere_detector import load_sidedetector_model, detect_side
from fetalbrain.alignment.align import align_scan
from path_literals import TEST_IMAGE_PATH


def test_load_sidedetector_model() -> None:
    model = load_sidedetector_model()
    assert isinstance(model, torch.nn.Module)


def test_detectside() -> None:
    example_scan, _ = read_image(TEST_IMAGE_PATH)
    side_model = load_sidedetector_model()

    # from atlas space
    aligned_atlas, _ = align_scan(example_scan, scale=False, to_atlas=True)
    pred_atl, probs_atl = detect_side(aligned_atlas, side_model, from_atlas=True)

    assert pred_atl == 0
    assert np.isclose(probs_atl, 0.99, atol=0.01)

    # from bean space
    aligned_im, _ = align_scan(example_scan, scale=False, to_atlas=False)
    pred, probs = detect_side(aligned_im, side_model, from_atlas=False)

    # small difference is expected but should be small
    assert (probs_atl.item() - probs.item()) < 0.01
