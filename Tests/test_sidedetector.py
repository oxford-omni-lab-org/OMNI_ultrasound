import torch
from pathlib import Path
from fetalbrain.utils import read_image, plot_midplanes
from fetalbrain.tedsnet_multi.hemisphere_detector import load_sidedetector_model, detect_side
from fetalbrain.alignment.align import align_scan
from fetalbrain.alignment.align import _get_atlastransform, transform_from_affine


TEST_IMAGE_PATH = Path("src/fetalbrain/alignment/test_data/06-5010_152days_0356.mha")


def test_load_sidedetector_model() -> None:
    model = load_sidedetector_model()
    assert isinstance(model, torch.nn.Module)


def test_detectside() -> None:
    example_scan, _ = read_image(TEST_IMAGE_PATH)

    # align to bean space
    aligned_im, params = align_scan(example_scan, scale=False, to_atlas=False)

    aligned_atlas, params = align_scan(example_scan, scale=False, to_atlas=True)

    # predict side
    model = load_sidedetector_model()
    side, certainty = detect_side(aligned_im, model)
    print(side)
    
    # we might get away with this if we also change hte plane that is selected
    atlas = torch.permute(aligned_atlas, (0, 1, 3, 2, 4))

    model = load_sidedetector_model()
    side, certainty2 = detect_side(atlas, model)
    print(certainty2)