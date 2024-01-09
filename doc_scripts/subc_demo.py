from fetalbrain.structural_segmentation.subcortical_segm import segment_scan_subc
from fetalbrain.utils import read_image, plot_planes_segm
from fetalbrain.alignment.align import align_scan
from fetalbrain.model_paths import EXAMPLE_IMAGE_PATH

# load an image
example_scan, _ = read_image(EXAMPLE_IMAGE_PATH)

# align the scan to the atlas space without scaling
aligned_scan, params = align_scan(example_scan, scale=False)

# perform segmentation on aligned image
segmentation, structure_names = segment_scan_subc(aligned_scan, connected_component=True)

# plot the segmentation
fig = plot_planes_segm(aligned_scan.squeeze(), segmentation.squeeze())
