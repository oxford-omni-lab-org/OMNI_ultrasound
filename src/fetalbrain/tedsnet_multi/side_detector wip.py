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
    segment_tedsall,
)
import os
from fetalbrain.utils import read_image, write_image, plot_midplanes
from fetalbrain.structural_segmentation.subcortical_segm import prepare_scan_segm
from fetalbrain.alignment.align import align_scan
from fetalbrain.tedsnet_multi.network.TEDS_Net import TEDS_Net
def setup_netw():
    net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False,num_classes=2)
    state_dict= torch.load("/home/sedm6226/Documents/Projects/US_analysis_package/src/fetalbrain/tedsnet_multi/network/FinalModel_sidedetection.pt")
    net.load_state_dict(state_dict)
    net.eval()
    return net

pathx = "/home/sedm6226/Documents/Projects/US_analysis_package/Original_Codes/SideDetection_Packaged_Maddy/Data/scans/07-10279_20130722_1263.nii.gz"


vol = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(pathx))) * 255
ss = 80
slice = np.expand_dims(vol[ss - 1 : ss + 2, :, :], 0)  # 3 channels
slice = torch.from_numpy(slice.astype(np.float32))  # move 3 to the front

device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")


net = setup_netw().to(device)
im = slice.to(device)

outputs = torch.sigmoid(net(im)).detach().cpu().numpy()
pred = np.argmax(outputs)

print(pred)
# 0 is left, 1 is right







TEST_IMAGE_PATH = Path("src/fetalbrain/alignment/test_data/06-5010_152days_0356.mha")
example_scan, _ = read_image(TEST_IMAGE_PATH)

aligned_scan, params = align_scan(example_scan, to_atlas=False)

# this is caused by maddy reading in the scan with simpleitk rather than nifty
aligned_scan_perm  = aligned_scan.permute(0, 1, 4, 3, 2)

ss = 80
slice = aligned_scan_perm[:, 0, ss - 1 : ss + 2, :, :] # 3 channels
outputs = torch.sigmoid(net(slice.to(device) * 255)).detach().cpu().numpy()


plot_midplanes(aligned_scan_perm.squeeze().numpy(), 'bean_per')

# so for an image, we need to align 
# aligned to bean, used for side detection
# aligned to atlas, used for segmentation