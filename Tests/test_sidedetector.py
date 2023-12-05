import torch
from pathlib import Path
from fetalbrain.utils import read_image
from fetalbrain.tedsnet_multi.hemisphere_detector import load_sidedetector_model, detect_side
from fetalbrain.alignment.align import align_scan

TEST_IMAGE_PATH = Path("src/fetalbrain/alignment/test_data/06-5010_152days_0356.mha")


def test_load_sidedetector_model() -> None:
    model = load_sidedetector_model()
    assert isinstance(model, torch.nn.Module)


def test_detectside() -> None:
    example_scan, _ = read_image(TEST_IMAGE_PATH)

    # align to bean space
    side_model = load_sidedetector_model()
    aligned_im, params = align_scan(example_scan, scale=False, to_atlas=False)
    pred, probs = detect_side(aligned_im, side_model, from_atlas=False)

    # get similar prediction from atlas
    aligned_atlas, params = align_scan(example_scan, scale=False, to_atlas=True)
    pred_atl, probs_atl = detect_side(aligned_atlas, side_model, from_atlas=True)

    # small difference is expected but should be small
    assert (probs_atl.item() - probs.item()) < 0.01
