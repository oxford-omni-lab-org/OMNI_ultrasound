import torch
from pathlib import Path
import os
import sys

sys.path.append("/home/sedm6226/Documents/Projects/US_analysis_package")
from src.alignment.align import load_alignment_model, align_to_atlas, prepare_scan  # noqa: E402
from src.utils import read_image, write_image  # noqa: E402
from src.structural_segmentation.segmentation_model import UNet  # noqa: E402


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TEST_IMAGE_PATH = Path("src/alignment/test_data/06-5010_152days_0356.mha")

example_scan, _ = read_image(TEST_IMAGE_PATH)
torch_scan = prepare_scan(example_scan)
align_model = load_alignment_model()

# align scan no scaling
aligned_scan, params = align_to_atlas(torch_scan, align_model, scale=False)


# segment subcortical structures
model_folder = "/mnt/data/Projects_Results/subcortical_segmentation/Final_Experiments2/Results/03_11_2021_26_SingleVolumesAligned_20training/run_0/"
model_path = os.path.join(model_folder, "modelcheckpoint_epoch_999_loss0.042063.tar")
net = torch.nn.DataParallel(UNet(1, 5, min_featuremaps=16, depth=5))
model = torch.load(model_path)
net.load_state_dict(model["model_state_dict"])
net.eval()

# because we load / save now with nii format, we need to permute the dimensions so that it matched the model's expected input
aligned_scan_per = aligned_scan.permute(0, 1, 4, 3, 2)
with torch.no_grad():
    output = net(aligned_scan_per*255)

output = torch.softmax(output, dim=1).permute(0, 1, 4, 3, 2)

path = Path('src/structural_segmentation/testdata')
write_path = path / "segmentation.nii.gz"

print(output.shape)
write_image(path / 'aligned_image.nii.gz', aligned_scan[0,0].cpu().numpy(), spacing=(0.6, 0.6, 0.6))

segm_map = output[0].argmax(dim=0).cpu().numpy()

write_image(write_path,segm_map, spacing=(0.6, 0.6, 0.6))


from src.structural_segmentation.subcortical_segm import  segment_subcortical, prepare_scan_segm
from src.structural_segmentation.subcortical_segm import compute_volume_segm
aligned_scan_prep = prepare_scan_segm(aligned_scan)


segm_map, key_maps  = segment_subcortical(aligned_scan_prep.permute((0,1, 4, 3, 2)), net)


import SimpleITK as sitk

segm_map = segm_map[0].cpu().numpy()

import numpy as np


def keep_largest_compoment(segm_map: np.ndarray) -> np.ndarray:

    conn_comp_segm = np.zeros_like(segm_map)

    # we assume 0 to be background so not connected component analysis
    for classx in torch.unique(segm_map)[1:]:
        single_class = np.where(segm_map == classx, 1, 0)

        sitk_im = sitk.GetImageFromArray(single_class)
        # assigns a unique label to each connected component
        labels_cc = sitk.ConnectedComponent(sitk_im, True)

        # relabels the components so that the largest component is 1
        labels_ordered = sitk.RelabelComponent(labels_cc)
        
        # get the largest connected component
        largest_cc = labels_ordered == 1

        # convert to numpy array
        largest_cc = sitk.GetArrayFromImage(largest_cc)

        # get the original class_value back
        largest_cc = largest_cc * classx

        conn_comp_segm += largest_cc
    
    return conn_comp_segm



