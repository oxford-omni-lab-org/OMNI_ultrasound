from pathlib import Path
from fetalbrain.alignment.align import align_scan
from fetalbrain.utils import read_image, plot_midplanes
from fetalbrain.model_paths import EXAMPLE_IMAGE_PATH

TEST_IMAGE_PATH = Path(EXAMPLE_IMAGE_PATH)
image, _ = read_image(TEST_IMAGE_PATH)

# align the scan to the atlas space
aligned_image, params = align_scan(image, to_atlas=True, scale=False)
plot_midplanes(aligned_image.numpy().squeeze(), 'Image aligned to atlas space')

# align the scan to the BEAN space
aligned_image, params = align_scan(image, to_atlas=False, scale=False)
plot_midplanes(aligned_image.numpy().squeeze(), 'Image aligned to BEAN space')
