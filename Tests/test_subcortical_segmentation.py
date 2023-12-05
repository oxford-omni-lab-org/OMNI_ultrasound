import doctest
import numpy as np
import torch
from pathlib import Path
from fetalbrain.structural_segmentation.subcortical_segm import (
    load_segmentation_model,
    prepare_scan_segm,
    segment_subcortical,
)
from fetalbrain.structural_segmentation.segmentation_model import UNet
from fetalbrain.utils import read_image
from fetalbrain.alignment.align import load_alignment_model, align_to_atlas, prepare_scan

#doctest.testmod()

TEST_IMAGE_PATH = Path("Tests/testdata/alignment/06-5010_152days_0356.mha")
REF_SEGMAP_PATH = Path("Tests/testdata/subcortical_segm/segmentation_ref.nii.gz")


def test_load_segmentation_model() -> None:
    model = load_segmentation_model()
    assert isinstance(model, torch.nn.DataParallel)
    assert isinstance(model.module, UNet)


def test_prepare_scan_segm() -> None:
    example_scan, _ = read_image(TEST_IMAGE_PATH)
    torch_scan = prepare_scan_segm(example_scan)
    assert torch_scan.shape == (1, 1, 160, 160, 160)

    # make sure it is scaled between 0 and 255
    assert torch.max(torch_scan) > 1


def test_segment_subcortical() -> None:
    # align scan
    example_scan, _ = read_image(TEST_IMAGE_PATH)
    torch_scan = prepare_scan(example_scan)
    align_model = load_alignment_model()

    # align scan no scaling
    aligned_scan, params = align_to_atlas(torch_scan, align_model, scale=False)

    # prepare scan for segmentation
    segm_model = load_segmentation_model()
    aligned_scan_prep = prepare_scan_segm(aligned_scan)
    multi_class, class_dict = segment_subcortical(aligned_scan_prep, segm_model)

    assert multi_class.shape == (1, 160, 160, 160)
    assert torch.max(multi_class) == 4

    # compare to a saved reference segmentation for this volume
    multi_class_np = multi_class[0].cpu().numpy()
    ref_segm, _ = read_image(REF_SEGMAP_PATH)

    assert np.all(multi_class_np == ref_segm)
