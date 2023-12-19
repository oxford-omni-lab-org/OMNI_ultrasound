from medpy.io import load
import numpy as np
import torch
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Literal
from fetalbrain.tedsnet_multi.teds_multi_segm import (
    load_tedsmulti_model,
    get_prior_shape_sa,
    generate_multiclass_prediction,
    segment_scan_tedsall,
    segment_tedsall,
)
import os
from fetalbrain.utils import read_image, write_image, plot_midplanes
from fetalbrain.tedsnet_multi.hemisphere_detector import load_sidedetector_model, detect_side
from fetalbrain.brain_extraction.extract import extract_scan_brain
from fetalbrain.alignment.align import align_scan, prepare_scan
from fetalbrain.tedsnet_multi.network.TEDS_Net import TEDS_Net



pathx = Path('/home/sedm6226/Documents/Projects/US_analysis_package/Original_Codes/Teds_All_Maddy/Data_utrechttest')
pathlist = pathx.glob('*.mha')

for pathnum in pathlist:
    example_scan, _ = read_image(pathnum)
    aligned_scan, params = align_scan(example_scan, to_atlas=True)
    write_image(pathnum.parent / (pathnum.stem + '_aligned.nii.gz'), aligned_scan.cpu().numpy().squeeze())
    tedssegm, keys = segment_scan_tedsall(aligned_scan)
    write_image(pathnum.parent/ (pathnum.stem + '_tedsmulti.nii.gz'), tedssegm.squeeze())

    whole_brain = extract_scan_brain(aligned_scan)
    write_image(pathnum.parent / (pathnum.stem + '_brainmask.nii.gz'), whole_brain.cpu().numpy().squeeze())











def setup_netw():
    net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False,num_classes=2)
    state_dict= torch.load("/home/sedm6226/Documents/Projects/US_analysis_package/src/model_weights/teds_segmentation/FinalModel_sidedetection.pt")
    net.load_state_dict(state_dict)
    net.eval()
    return net

pathx = "/home/sedm6226/Documents/Projects/US_analysis_package/Original_Codes/SideDetection_Packaged_Maddy/Data/scans/07-10279_20130722_1263.nii.gz"


vol, spacing = read_image(Path(pathx))
aligned_scan, _parms = align_scan(vol, scale=False, to_atlas=False)
ss = 80

# extract slice
slicex = aligned_scan[0,0, :, :, ss - 1 : ss + 2].permute(2, 1, 0)



device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")
net = setup_netw().to(device)



output_ban = detect_side(aligned_scan.to(device), net, from_atlas=False)
aligned_scan, _parms = align_scan(vol, scale=False, to_atlas=True)
output_atlas = detect_side(aligned_scan.to(device), net, from_atlas=True)

im = slice.to(device)

outputs = torch.sigmoid(net(slicex.to(device).unsqueeze(0))).detach().cpu().numpy()
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