import json
import torch
import numpy as np
from pathlib import Path
import sys
from typing import Optional

sys.path.append("/home/sedm6226/Documents/Projects/US_analysis_package")

from src.utils import read_image, write_image  # noqa: E402
from src.alignment.kelluwen_transforms import generate_affine, apply_affine  # noqa: E402
from src.alignment.fBAN_v1 import AlignModel  # noqa: E402
from src.alignment.utils_alignment import plot_midplanes  # noqa: E402
from src.alignment.align import load_alignment_model, align_scan, prepare_scan, unalign_scan
from src.alignment.kelluwen_transforms import apply_affine  # noqa: E402


model = load_alignment_model()
test_path = Path("src/alignment/test_data/06-5010_152days_0356.mha")

image, spacing = read_image(test_path)
example_scan = prepare_scan(image)




# Plot for initial orientation ------------------------------------------------------------------------------


fig = plot_midplanes(image, "Input original")
fig.savefig("src/alignment/test_data/original_scan.png")


example_scan = prepare_scan(image)
example_prediction_translation, example_prediction_rotation, example_prediction_scaling = model(example_scan)

example_scan_permutation = example_scan.permute((0, 1, 4, 3, 2))
example_prediction_translation_2, example_prediction_rotation_2, example_prediction_scaling_2 = model(example_scan_permutation)



example_transform = generate_affine(
    parameter_translation=example_prediction_translation * 160,
    parameter_rotation=example_prediction_rotation,
    parameter_scaling=example_prediction_scaling,
    type_rotation="quaternions",
    transform_order="srt",
)
example_scan_aligned = apply_affine(example_scan, example_transform)
fig = plot_midplanes(example_scan_aligned.squeeze().cpu().numpy(), "Aligned scan Original")
fig.savefig("src/alignment/test_data/aligned_scan_original.png")

# Plot for permuted orientation ------------------------------------------------------------------------------

example_scan_perms = example_scan.permute((0, 1, 3, 4, 2))



fig = plot_midplanes(example_scan_perms.squeeze().cpu().numpy(), "Input permuted")
fig.savefig("src/alignment/test_data/input_permuted.png")


example_prediction_translation_2, example_prediction_rotation_2, example_prediction_scaling_2 = model(
    example_scan_perms
)

example_transform_2 = generate_affine(
    parameter_translation=example_prediction_translation_2 * 160,
    parameter_rotation=example_prediction_rotation_2,
    parameter_scaling=example_prediction_scaling_2,
    type_rotation="quaternions",
    transform_order="srt",
)
example_scan_aligned = apply_affine(example_scan_perms, example_transform_2)
fig = plot_midplanes(example_scan_aligned.squeeze().cpu().numpy(), "Aligned scan permuted")
fig.savefig("src/alignment/test_data/aligned_scan_permuted.png")

