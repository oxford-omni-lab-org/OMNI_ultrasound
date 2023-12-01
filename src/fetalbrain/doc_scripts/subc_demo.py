from pathlib import Path
from fetalbrain.structural_segmentation.subcortical_segm import segment_scan_subc
from fetalbrain.utils import read_image, plot_planes_segm
from fetalbrain.alignment.align import align_scan

# load an image
TEST_IMAGE_PATH = Path("src/fetalbrain/alignment/test_data/06-5010_152days_0356.mha")
example_scan, _ = read_image(TEST_IMAGE_PATH)

# align the scan to the atlas space without scaling
aligned_scan, params = align_scan(example_scan, scale=False)

# perform segmentation on aligned image
segmentation, structure_names = segment_scan_subc(aligned_scan, connected_component=True)

# plot the segmentation
fig = plot_planes_segm(aligned_scan.squeeze(), segmentation.squeeze())
