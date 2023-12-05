from pathlib import Path
from fetalbrain.utils import read_image, write_image
from fetalbrain.alignment.align import align_scan
from fetalbrain.structural_segmentation.subcortical_segm import segment_scan_subc
from fetalbrain.tedsnet_multi.teds_multi_segm import segment_scan_tedsall

TEST_IMAGE_PATH = Path("src/fetalbrain/alignment/test_data/06-5010_152days_0356.mha")

example_scan, _ = read_image(TEST_IMAGE_PATH)


# Start with alignment to atlas space
aligned_scan, params = align_scan(example_scan, scale=False, to_atlas=True)

# Perform subcortical segmentation
subc_segm, keymaps = segment_scan_subc(aligned_scan, connected_component=True)

# Use TEDSnet to do all structure segmentations
allstructure_segm, keymaps = segment_scan_tedsall(aligned_scan)


# Write out the results in the aligned orientation
write_image(Path("src/fetalbrain/alignment/test_data/aligned_scan.nii.gz"), aligned_scan.squeeze().numpy() * 255)
write_image(Path("src/fetalbrain/alignment/test_data/subcortical_segm.nii.gz"), subc_segm.squeeze())
write_image(Path("src/fetalbrain/alignment/test_data/allstructure_segm.nii.gz"), allstructure_segm.squeeze())
