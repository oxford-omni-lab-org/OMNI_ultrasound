from pathlib import Path
from fetalbrain.utils import read_image, write_image
from fetalbrain.alignment.align import align_scan
from fetalbrain.structural_segmentation.subcortical_segm import segment_scan_subc, compute_volume_segm
from fetalbrain.tedsnet_multi.teds_multi_segm import segment_scan_tedsall
from fetalbrain.brain_extraction.extract import extract_scan_brain
from fetalbrain.model_paths import EXAMPLE_IMAGE_PATH

example_scan, _ = read_image(EXAMPLE_IMAGE_PATH)

# Start with alignment to atlas space
aligned_scan, params = align_scan(example_scan, scale=False, to_atlas=True)

# Perform subcortical segmentation
subc_segm, subc_keys = segment_scan_subc(aligned_scan, connected_component=True)

# Use TEDSnet to do all structure segmentations
allstructure_segm, multi_keys = segment_scan_tedsall(aligned_scan)

# Extract the brain
brain_mask, brain_key = extract_scan_brain(aligned_scan)

# Write out the results in the aligned orientation
write_image(Path("aligned_scan.nii.gz"), aligned_scan.squeeze().numpy())
write_image(Path("subcortical_segm.nii.gz"), subc_segm.squeeze(), segm=True)
write_image(Path("allstructure_segm.nii.gz"), allstructure_segm.squeeze(), segm=True)
write_image(Path("brain_mask.nii.gz"), brain_mask.squeeze(), segm=True)

# compute volumes of the segmentation masks
volume_dict = compute_volume_segm(subc_segm, subc_keys, spacing=(0.6, 0.6, 0.6))
print(volume_dict)
