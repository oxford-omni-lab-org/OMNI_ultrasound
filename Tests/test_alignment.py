import torch
from PIL import Image
import numpy as np
from fetalbrain.utils import read_image, write_image
from fetalbrain.alignment.fBAN_v1 import AlignmentModel
from fetalbrain.alignment.align import (
    load_alignment_model,
    prepare_scan,
    align_to_bean,
    transform_from_params,
    transform_from_affine,
    _get_transform_to_atlasspace,
    _get_atlastransform_itksnap,
    align_to_atlas,
)
from path_literals import TEST_IMAGE_PATH, TEST_ALIGNMENT_PATH, TEMP_SAVEPATH


def compare_threshold(scan1: torch.Tensor, scan2: torch.Tensor, threshold: float = 1.0) -> float:
    no_pixels = 160 * 160 * 160
    percentage_equal = torch.count_nonzero(torch.abs(scan1 - scan2) <= threshold) / (no_pixels) * 100
    return percentage_equal.item()


def test_load_alignment_model() -> None:
    model = load_alignment_model()
    assert isinstance(model, AlignmentModel)


def test_prepare_scan() -> None:
    example_scan, _ = read_image(TEST_IMAGE_PATH)
    torch_scan = prepare_scan(example_scan)

    # expecting it to be a tensor between 0 and 255
    assert torch.min(torch_scan) >= 0.0
    assert torch.max(torch_scan) > 1.0
    assert torch.max(torch_scan) <= 255.0
    assert torch_scan.shape == (1, 1, 160, 160, 160)


def test_align_scan_bean() -> None:
    """compare the alignment with a reference image for BEAN aligned images"""
    example_scan, spacing = read_image(TEST_IMAGE_PATH)
    torch_scan = prepare_scan(example_scan)
    model = load_alignment_model()

    aligned_scan, params = align_to_bean(torch_scan, model)

    # verify the scan is of the same shape as before
    assert aligned_scan.shape == torch_scan.shape
    assert torch.min(aligned_scan) >= 0.0
    assert torch.max(aligned_scan) <= 255.0

    assert "scaling" in params.keys()
    assert "rotation" in params.keys()
    assert "translation" in params.keys()

    assert params["scaling"].shape == (1, 3)
    assert params["rotation"].shape == (1, 4)
    assert params["translation"].shape == (1, 3)

    # verify that the aligned scan is the same as before
    aligned_scan_np = aligned_scan.squeeze().cpu().numpy()
    ref_frame = np.array(Image.open(TEST_ALIGNMENT_PATH / "aligned_axial_ref.png"))
    new_frame = np.uint8(aligned_scan_np[:, :, 80])

    # test that they differ no more than 1 pixel
    assert np.allclose(ref_frame, new_frame, atol=1), "The aligned image is too different from the reference image"

    # to save a new reference image use:
    # pil_img = Image.fromarray(np.uint8(aligned_scan_np[:, :, 80]), mode="L")
    # pil_img.save(TEST_ALIGNMENT_PATH / "aligned_axial_ref.png", "png")


def test_align_scan_atlas() -> None:
    """compare the alignment with a reference image for atlas aligned images"""
    example_scan, spacing = read_image(TEST_IMAGE_PATH)
    torch_scan = prepare_scan(example_scan)
    model = load_alignment_model()

    aligned_scan, params = align_to_atlas(torch_scan, model)

    # verify the scan is of the same shape as before
    assert aligned_scan.shape == torch_scan.shape
    assert torch.min(aligned_scan) >= 0.0
    assert torch.max(aligned_scan) <= 255.0

    assert "scaling" in params.keys()
    assert "rotation" in params.keys()
    assert "translation" in params.keys()

    assert params["scaling"].shape == (1, 3)
    assert params["rotation"].shape == (1, 4)
    assert params["translation"].shape == (1, 3)

    # verify that the aligned scan is the same as before
    aligned_scan_np = aligned_scan.squeeze().cpu().numpy()
    ref_frame = np.array(Image.open(TEST_ALIGNMENT_PATH / "aligned_axial_ref_atlas.png"))
    new_frame = np.uint8(aligned_scan_np[:, :, 80])

    # test that they differ no more than 1 pixel
    assert np.allclose(ref_frame, new_frame, atol=1), "The aligned image is too different from the reference image"

    # to save a new reference image use:
    # pil_img = Image.fromarray(np.uint8(aligned_scan_np[:, :, 80]), mode="L")
    # pil_img.save(TEST_ALIGNMENT_PATH / "aligned_axial_ref_atlas.png", "png")


def test_align_from_params() -> None:
    example_scan, _ = read_image(TEST_IMAGE_PATH)
    torch_scan = prepare_scan(example_scan)
    model = load_alignment_model()

    # Direct alignment with BEAN
    aligned, params = align_to_bean(torch_scan, model)

    # Use alignment parameters to align (clamping required after)
    aligned_from_params = transform_from_params(
        torch_scan, rotation=params["rotation"], translation=params["translation"], scaling=params["scaling"]
    )
    aligned_from_params = torch.clamp(aligned_from_params, 0, 255)

    # ensure they are the same
    assert torch.all(aligned == aligned_from_params)

    # set certain parameters to default values
    aligned_from_params = transform_from_params(
        torch_scan, translation=params["translation"], scaling=params["scaling"]
    )
    aligned_from_params = transform_from_params(torch_scan, rotation=params["rotation"], scaling=params["scaling"])
    aligned_from_params = transform_from_params(
        torch_scan, rotation=params["rotation"], translation=params["translation"]
    )


def test_unalign_scan_bean() -> None:
    """This test tests that a scan can be aligned and then unaligned to the original image,
    resulting in the same image."""
    example_scan, spacing = read_image(TEST_IMAGE_PATH)
    torch_scan = prepare_scan(example_scan)
    model = load_alignment_model()

    aligned_image, _, transform = align_to_bean(torch_scan, model, return_affine=True, scale=False)
    unaligned_im = transform_from_affine(aligned_image, transform.inverse())

    # verify that the aligned scan is the same as before
    perc_equal = compare_threshold(torch_scan, unaligned_im, threshold=5.0)

    # 95% of the pixels should be within 5 of the original value, the exact thresholds are arbitrary
    assert perc_equal > 95


def test_unalign_scan_atlas() -> None:
    """This test tests that a scan can be aligned and then unaligned to the original image,
    resulting in the same image."""
    example_scan, spacing = read_image(TEST_IMAGE_PATH)
    torch_scan = prepare_scan(example_scan)
    model = load_alignment_model()

    aligned_image, _, transform = align_to_atlas(torch_scan, model, return_affine=True, scale=False)
    unaligned_im = transform_from_affine(aligned_image, transform.inverse())

    # verify that the aligned scan is the same as before
    perc_equal = compare_threshold(torch_scan, unaligned_im, threshold=5.0)

    # 95% of the pixels should be within 5 of the original value, the exact thresholds are arbitrary
    assert perc_equal > 95


def test_scaling_twosteps_bean() -> None:
    """This function test whether applying first alignment without scaling, and then applying the scaling seperately
    gives the same result as applying the alignment + scaling in one step."""
    example_scan, spacing = read_image(TEST_IMAGE_PATH)
    torch_scan = prepare_scan(example_scan)
    model = load_alignment_model()

    # 1 step approach
    aligned_scan, params = align_to_bean(torch_scan, model, scale=True)

    # 2 step approach
    aligned_noscale, _ = align_to_bean(torch_scan, model, scale=False)
    aligned_twostep = transform_from_params(aligned_noscale, scaling=params["scaling"])

    # verify that the aligned scan is the same as before
    perc_equal = compare_threshold(aligned_scan, aligned_twostep, threshold=5.0)
    assert perc_equal > 98


def test_scaling_twosteps_atlas() -> None:
    """This function test whether applying first alignment without scaling, and then applying the scaling seperately
    gives the same result as applying the alignment + scaling in one step."""
    example_scan, spacing = read_image(TEST_IMAGE_PATH)
    torch_scan = prepare_scan(example_scan)
    model = load_alignment_model()

    # 1 step approach
    aligned_scan, params = align_to_atlas(torch_scan, model, scale=True)

    # 2 step approach
    aligned_noscale, _ = align_to_atlas(torch_scan, model, scale=False)
    aligned_twostep = transform_from_params(aligned_noscale, scaling=params["scaling"])

    # verify that the aligned scan is the same as before
    perc_equal = compare_threshold(aligned_scan, aligned_twostep, threshold=5.0)
    assert perc_equal > 98


def test_permutations() -> None:
    """This functions tests whether permuting the axis results in the same alignment.
    For a permutation of two axis, the third one flips so therefore the result also needs to be flipped.

    A similar test can be made where apply the flipping to the input (after permution).
    This should also result in the same/similar alignment.
    """
    example_scan, _ = read_image(TEST_IMAGE_PATH)
    torch_scan = prepare_scan(example_scan)
    model = load_alignment_model()

    # align scan, and permuted scan
    aligned_scan, _ = align_to_bean(torch_scan, model)
    aligned_scan_perm, _ = align_to_bean(torch.flip(torch_scan.permute(0, 1, 4, 3, 2), [3]), model)

    # larger margins because stochasticity of model results in larger differences (i.e. not only interpolation)
    perc_equal = compare_threshold(aligned_scan, aligned_scan_perm, threshold=10.0)
    assert perc_equal > 80


def test_get_atlastransform() -> None:
    """test function to assert the generated atlas transformations is correct"""
    atlas_transform = _get_atlastransform_itksnap()

    assert atlas_transform.shape == (1, 4, 4)

    translation = atlas_transform[0, :3, 3]
    assert torch.allclose(translation, torch.tensor([8.8243, 0.2393, 5.1726]), atol=1e-4)

    rotation = atlas_transform[0, :3, :3]
    assert torch.allclose(
        rotation,
        torch.tensor([[0.9929, 0.0120, -0.1181], [-0.0156, 0.9994, -0.0303], [0.1176, 0.0320, 0.9925]]),
        atol=1e-4,
    )


def test_get_transform_to_atlasspace() -> None:
    full_atlasspace_transform = _get_transform_to_atlasspace()
    assert full_atlasspace_transform.shape == (1, 4, 4)

    translation = full_atlasspace_transform[0, :3, 3]
    assert torch.allclose(translation, torch.tensor([8.8243, 0.2393, 5.1726]), atol=1e-4)

    rotation = full_atlasspace_transform[0, :3, :3]
    assert torch.allclose(
        rotation,
        torch.tensor([[-0.0120, 0.9929, -0.1181], [-0.9994, -0.0156, -0.0303], [-0.0320, 0.1176, 0.9925]]),
        atol=1e-4,
    )


def test_consistency() -> None:
    example_scan, spacing = read_image(TEST_IMAGE_PATH)
    torch_scan = prepare_scan(example_scan)
    model = load_alignment_model()

    # aligned image
    aligned_scan, params = align_to_atlas(torch_scan, model, scale=False)
    write_image(TEMP_SAVEPATH / "aligned.nii.gz", aligned_scan.numpy().squeeze(), spacing=(0.6, 0.6, 0.6))

    # reload and align again
    example_reloaded, _ = read_image(TEMP_SAVEPATH / "aligned.nii.gz")
    example_reloaded_torch = prepare_scan(example_reloaded)
    aligned_scan_reload, params = align_to_atlas(example_reloaded_torch, model, scale=False)

    # verify that they are similar
    perc_equal = compare_threshold(aligned_scan, aligned_scan_reload, threshold=10.0)
    assert perc_equal > 75
