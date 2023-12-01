from pathlib import Path
from fetalbrain.alignment.align import align_scan
from fetalbrain.utils import read_image, plot_midplanes

TEST_IMAGE_PATH = Path("src/fetalbrain/alignment/test_data/06-5010_152days_0356.mha")
image, _ = read_image(TEST_IMAGE_PATH)

# align the scan to the atlas space
aligned_image, params = align_scan(image, to_atlas=True, scale=False)
plot_midplanes(aligned_image.numpy().squeeze(), 'Image aligned to atlas space')

# align the scan to the BEAN space
aligned_image, params = align_scan(image, to_atlas=False, scale=False)
plot_midplanes(aligned_image.numpy().squeeze(), 'Image aligned to BEAN space')
