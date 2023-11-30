from medpy.io import load
import numpy as np
import torch
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Literal
from fetalbrain.tedsnet_multi.plain_segmenter import (
    load_tedsmulti_model,
    get_prior_shape_sa,
    generate_multiclass_prediction,
    segment_scan_tedsall,
    segment_tedsall
)
from fetalbrain.utils import read_image, write_image
from fetalbrain.structural_segmentation.subcortical_segm import prepare_scan_segm
from fetalbrain.alignment.align import align_scan
from fetalbrain.tedsnet_multi.network.TEDS_Net import TEDS_Net

TEST_PATH = "/home/sedm6226/Documents/Projects/US_analysis_package/src/fetalbrain/tedsnet_multi/testdata/scans/02-1612_154days_0757.mha"


## maddy's test set volume
x, _ = read_image(Path(TEST_PATH))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

segm_model = load_tedsmulti_model().to(device)
scan = prepare_scan_segm(x).permute(0, 1, 4, 3, 2).to(device)
multiclass, keys = segment_tedsall(scan, segm_model, side="l")

filepath = "/home/sedm6226/Documents/Projects/US_analysis_package/src/fetalbrain/tedsnet_multi/testdata/segs/02-1612_154days_0757_new4.nii.gz"
write_image(Path(filepath), multiclass)




# alignment -> tedsnet
TEST_IMAGE_PATH = Path("src/fetalbrain/alignment/test_data/06-5010_152days_0356.mha")
example_scan, _ = read_image(TEST_IMAGE_PATH)
aligned_scan, params = align_scan(example_scan)
tedssegm, keys = segment_scan_tedsall(aligned_scan.to(device))

filepath = "/home/sedm6226/Documents/Projects/US_analysis_package/src/fetalbrain/tedsnet_multi/testdata/segs/alignedvol_segm.nii.gz"
write_image(Path(filepath), tedssegm)


filepath_im = "/home/sedm6226/Documents/Projects/US_analysis_package/src/fetalbrain/tedsnet_multi/testdata/segs/alignedvol.nii.gz"
write_image(Path(filepath_im), aligned_scan.numpy().squeeze() * 255)
