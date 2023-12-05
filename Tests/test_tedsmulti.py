import torch
from pathlib import Path
from typing import Optional
import numpy as np
from fetalbrain.utils import read_image, plot_planes_segm
from fetalbrain.alignment.align import align_scan
from fetalbrain.tedsnet_multi.teds_multi_segm import get_prior_shape_sa, segment_scan_tedsall, load_tedsmulti_model

TEST_IMAGE_PATH = Path("Tests/testdata/alignment/06-5010_152days_0356.mha")


def test_load_tedsmulti_model(model_path: Optional[Path] = None) -> None:
    model = load_tedsmulti_model(model_path)
    assert isinstance(model, torch.nn.Module)


def test_get_prior_shape_sa() -> None:
    prior_shape = get_prior_shape_sa(sd="l")
    assert prior_shape.shape == (10, 160, 160, 160)

    prior_shape = get_prior_shape_sa(sd="r")
    assert prior_shape.shape == (10, 160, 160, 160)


def test_segment_scan_tedsall():
    example_scan, _ = read_image(TEST_IMAGE_PATH)
    aligned_scan, params = align_scan(example_scan, to_atlas=True)
    tedssegm, keys = segment_scan_tedsall(aligned_scan)

    assert tedssegm.shape == (160, 160, 160)
    assert np.max(tedssegm) == 10

    fig = plot_planes_segm(aligned_scan.squeeze(), tedssegm)
