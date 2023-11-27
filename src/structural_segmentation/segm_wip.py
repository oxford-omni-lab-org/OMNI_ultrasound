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
