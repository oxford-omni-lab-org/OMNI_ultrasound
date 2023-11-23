import json
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append("/home/sedm6226/Documents/Projects/US_analysis_package")

from src.utils import read_image, write_image  # noqa: E402
from src.alignment.kelluwen_transforms import generate_affine, apply_affine  # noqa: E402
from src.alignment.fBAN_v1 import AlignModel  # noqa: E402
from src.alignment.utils_alignment import (
    plot_midplanes,
    realign_volume,
    transform_pytorch,
    torchparams_to_transform,
)  # noqa: E402
from src.alignment.align import load_alignment_model, get_transform_to_atlasspace, align_scan, prepare_scan, unalign_scan
from src.alignment.kelluwen_transforms import apply_affine, deconstruct_affine, generate_affine  # noqa: E402


def get_transform_to_atlas():
    params_to_atlas = json.load(open("src/alignment/config/25wks_Atlas(separateHems)_mean_warped.json", "r"))
    eu_params = params_to_atlas["eu_param"]
    eu_param_to_atlas = torch.Tensor(eu_params).reshape(1, -1)
    tr_params = params_to_atlas["tr_param"]
    tr_param_to_atlas = torch.Tensor(tr_params).reshape(1, -1)
    sc_param_to_atlas = torch.Tensor([1, 1, 1]).reshape(1, -1)
    #atlas_transform = torchparams_to_transform(eu_param_to_atlas, tr_param_to_atlas, sc_param_to_atlas).inverse()

    # the negative, likely going from left to right handed coordinate system
    atlas_transform = generate_affine(
        tr_param_to_atlas, -eu_param_to_atlas, sc_param_to_atlas, type_rotation="euler_zyx", transform_order="trs"
    )
    #assert torch.all(atlas_transform == atlas_t2)
    return atlas_transform.inverse()


def reorient(aligned_scan, write_path="aligned_atlas_perms.nii.gz"):
    atlas_permflip = torch.flip(torch.permute(aligned_scan, dims=(0, 1, 4, 2, 3)), dims=[2])
    atlas_transform = get_transform_to_atlas()
    scan_atlas = transform_pytorch(atlas_permflip, atlas_transform, mode="bilinear")
    scan = np.transpose(scan_atlas.squeeze().numpy(), axes=(1, 0, 2))[:, ::-1]
    write_image(
        Path("src/alignment/test_data/") / write_path,
        scan,
        spacing=spacing,
    )
    return scan


def reorient_1step(aligned_scan, write_path="aligned_atlas_1step.nii.gz"):
    perm_matrix_1 = torch.tensor([[0, 0, -1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=torch.float32)

    atlas_transform = get_transform_to_atlas()

    perm_matrix_2 = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=torch.float32)

    total_affine = perm_matrix_1.unsqueeze(0) @ atlas_transform @ perm_matrix_2.unsqueeze(0)

    atlas_transformed = transform_pytorch(aligned_scan, total_affine, mode="bilinear").squeeze().numpy()

    write_image(
        Path("src/alignment/test_data/") / write_path,
        atlas_transformed,
        spacing=spacing,
    )

    return atlas_transformed


def align_kelluwen(aligned_scan, total_affine):
    total_affine[0, :3, 3] = total_affine[0, :3, 3] * 80

    # this is equivalent to applying the transformation matrix with changed orientation for only transl (i.e. abc -> cba for transl)
    aligned_perm2 = torch.permute(aligned_scan, dims=(0, 1, 4, 3, 2))
    atlas_transformed = apply_affine(
        aligned_perm2, total_affine.inverse(), type_resampling="bilinear", type_origin="centre"
    )
    atlas_transformed_2 = torch.permute(atlas_transformed, dims=(0, 1, 4, 3, 2))

    return atlas_transformed_2


def align_transform_pytorch(aligned_scan, total_affine):
    return transform_pytorch(aligned_scan, total_affine, mode="bilinear")


def align_kelluwen_direct(aligned_scan, total_affine):
    perm_matrix_1 = torch.tensor(
        [[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=torch.float32
    ).unsqueeze(0)
    atlas_direct = apply_affine(
        aligned_scan,
        perm_matrix_1 @ total_affine.inverse() @ perm_matrix_1.inverse(),
        type_resampling="bilinear",
        type_origin="centre",
    )
    return atlas_direct


def transform_to_atlas(aligned_scan):
    # the minus sign in the first permutation is required as we permute two axis
    first_perm = torch.tensor([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=torch.float32).unsqueeze(
        0
    )

    # the second flips the hemisphere, this one is not strictly required, check for consistency in data
    second_perm = torch.tensor(
        [[0, 1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=torch.float32
    ).unsqueeze(0)

    # atlas trnasform, multiply by 80
    atlas_transform = get_transform_to_atlas()
    atlas_transform[0, :3, 3] = atlas_transform[0, :3, 3] * 80

    # total transformation from felipe to atlas orientation
    total_transformation_2 = second_perm @ atlas_transform.inverse() @ first_perm

    # final function
    atlas_orientation = apply_affine(
        aligned_scan,
        total_transformation_2,
        type_resampling="bilinear",
        type_origin="centre",
    )
    return atlas_orientation


model = load_alignment_model()
test_path = Path("src/alignment/test_data/06-5010_152days_0356.mha")

image, spacing = read_image(test_path)
example_scan = prepare_scan(image)


aligned_scan, params = align_scan(example_scan, model, scale=False)

# assert that doing it in one step gives same results as doing permutations
scan_perms = reorient(aligned_scan, write_path="aligned_atlas_perms.nii.gz")
scan_1step = reorient_1step(aligned_scan, write_path="aligned_atlas_1step.nii.gz")
assert np.all(np.isclose(scan_perms, scan_1step, atol=0.001))


# total affine matrix
perm_matrix_1 = torch.tensor([[0, 0, -1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=torch.float32)
atlas_transform = get_transform_to_atlas()
perm_matrix_2 = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=torch.float32)
total_affine = perm_matrix_1.unsqueeze(0) @ atlas_transform @ perm_matrix_2.unsqueeze(0)


atlas_orien_torch = align_transform_pytorch(aligned_scan, total_affine)
atlas_orien_kelluwen = align_kelluwen(aligned_scan, total_affine)
atlas_direct = align_kelluwen_direct(aligned_scan, total_affine)

# only count where both are above zero (overcome tiny border artefacts)
mask = (1 * (atlas_orien_torch * atlas_orien_kelluwen) > 0).squeeze().numpy()
assert np.all(
    np.isclose(atlas_orien_torch.squeeze().numpy() * mask, atlas_orien_kelluwen.squeeze().numpy() * mask, atol=0.001)
)


mask = (1 * (atlas_direct * atlas_orien_kelluwen) > 0).squeeze().numpy()
assert np.all(
    np.isclose(atlas_direct.squeeze().numpy() * mask, atlas_orien_kelluwen.squeeze().numpy() * mask, atol=0.001)
)


atlas_complete = transform_to_atlas(aligned_scan)

mask = (1 * (atlas_complete * atlas_orien_kelluwen) > 0).squeeze().numpy()
assert np.all(
    np.isclose(atlas_complete.squeeze().numpy() * mask, atlas_orien_kelluwen.squeeze().numpy() * mask, atol=0.001)
)



# -----------------------------------------------------------------------------------------------
params_to_atlas = json.load(open("src/alignment/config/25wks_Atlas(separateHems)_mean_warped.json", "r"))
eu_params = params_to_atlas["eu_param"]
eu_param_to_atlas = torch.Tensor(eu_params).reshape(1, -1)
tr_params = params_to_atlas["tr_param"]
tr_param_to_atlas = torch.Tensor(tr_params).reshape(1, -1)
sc_param_to_atlas = torch.Tensor([1, 1, 1]).reshape(1, -1)


# the (-) determines counterclockwise rotation, this is somehow differently defined
atlas_transform = torchparams_to_transform(eu_param_to_atlas, tr_param_to_atlas, sc_param_to_atlas)
atlas_t2 = generate_affine(
    tr_param_to_atlas, -eu_param_to_atlas, sc_param_to_atlas, type_rotation="euler_zyx", transform_order="trs"
)
assert torch.all(atlas_transform == atlas_t2)


# -----------------------------------------------------------------------
TEST_IMAGE_PATH = Path("src/alignment/test_data/06-5010_152days_0356.mha")

example_scan, spacing = read_image(TEST_IMAGE_PATH)
torch_scan = prepare_scan(example_scan)
model = load_alignment_model()

# align scan (has to be scale = false to make transform to atlasspace work)
aligned_scan, params, affine = align_scan(torch_scan, model, return_affine=True, scale=False)

# get the transformation to atlas space
atlas_transform = get_transform_to_atlasspace()

# 1 and 2 step approach
aligned_atlas = apply_affine(aligned_scan, atlas_transform)








write_image(
    Path("src/alignment/test_data/") / "aligned_scan.nii.gz",
    aligned_scan.squeeze().numpy(),
    spacing=spacing,
)
write_image(
    Path("src/alignment/test_data/") / "atlas_direct_2.nii.gz",
    scan_perms.squeeze().numpy(),
    spacing=spacing,
)


# to do write image between 0 and 255, not 0 and 1
# combine permutation matrices into as few as possible to make it more transparent how to go from felipe to atlas orientation

# Plot for initial orientation ------------------------------------------------------------------------------


fig = plot_midplanes(image, "Input original")
fig.savefig("src/alignment/test_data/original_scan.png")


example_scan = prepare_scan(image)
example_prediction_translation, example_prediction_rotation, example_prediction_scaling = model(example_scan)

example_scan_permutation = example_scan.permute((0, 1, 4, 3, 2))
example_prediction_translation_2, example_prediction_rotation_2, example_prediction_scaling_2 = model(
    example_scan_permutation
)


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
