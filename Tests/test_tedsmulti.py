import torch
from pathlib import Path
from typing import Optional
import numpy as np
from fetalbrain.utils import read_image
from fetalbrain.alignment.align import align_scan
from fetalbrain.tedsnet_multi.teds_multi_segm import get_prior_shape_sa, segment_scan_tedsall, load_tedsmulti_model
from path_literals import TEST_IMAGE_PATH, TEST_MULTITEDS_PATH


def test_load_tedsmulti_model(model_path: Optional[Path] = None) -> None:
    model = load_tedsmulti_model(model_path)
    assert isinstance(model, torch.nn.Module)


def test_get_prior_shape_sa() -> None:
    prior_shape = get_prior_shape_sa(sd=0)
    assert prior_shape.shape == (10, 160, 160, 160)
    assert torch.all(prior_shape[:, :, 85:] == 0)

    prior_shape = get_prior_shape_sa(sd=1)
    assert prior_shape.shape == (10, 160, 160, 160)
    assert torch.all(prior_shape[:, :, :75] == 0)


def test_segment_scan_tedsall() -> None:
    example_scan, _ = read_image(TEST_IMAGE_PATH)
    aligned_scan, params = align_scan(example_scan, to_atlas=True)
    tedssegm, keys = segment_scan_tedsall(aligned_scan)

    assert tedssegm.shape == (160, 160, 160)
    assert np.max(tedssegm) == 10

    ref_segmpath = TEST_MULTITEDS_PATH / "ref_tedsmulti.nii.gz"
    ref_segm, _ = read_image(ref_segmpath)

    assert ref_segm.shape == tedssegm.shape
    assert np.allclose(ref_segm, tedssegm, atol=1e-4)
