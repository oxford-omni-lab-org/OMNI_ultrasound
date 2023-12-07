from pathlib import Path
import torch
import doctest
from fetalbrain.utils import read_image
from fetalbrain.alignment.fBAN_v1 import AlignmentModel
from fetalbrain.alignment.align import (
    load_alignment_model,
    prepare_scan,
    align_to_bean,
    transform_from_params,
    transform_from_affine,
    _get_transform_to_atlasspace,
    align_to_atlas,
)  # noqa: E402
from fetalbrain.utils import write_image, plot_midplanes
from fetalbrain.alignment.kelluwen_transforms import apply_affine

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TEST_IMAGE_PATH = Path("Tests/testdata/example_image.nii.gz")
TEMP_SAVE_PATH = Path("Tests/testdata/alignment/temp")

# doctest.testmod()


def test_load_alignment_model() -> None:
    model = load_alignment_model()
    assert isinstance(model, AlignmentModel)


def test_prepare_scan() -> None:
    example_scan, _ = read_image(TEST_IMAGE_PATH)
    torch_scan = prepare_scan(example_scan)
    assert torch_scan.shape == (1, 1, 160, 160, 160)


def test_align_scan() -> None:
    example_scan, spacing = read_image(TEST_IMAGE_PATH)
    torch_scan = prepare_scan(example_scan)
    model = load_alignment_model()

    aligned_scan, params = align_to_bean(torch_scan, model)

    # verify the scan is of the same shape as before
    assert aligned_scan.shape == torch_scan.shape

    assert "scaling" in params.keys()
    assert "rotation" in params.keys()
    assert "translation" in params.keys()

    assert params["scaling"].shape == (1, 3)
    assert params["rotation"].shape == (1, 4)
    assert params["translation"].shape == (1, 3)

    write_image(
        TEMP_SAVE_PATH / "aligned_scan.nii.gz",
        aligned_scan.squeeze().cpu().numpy(),
        spacing=spacing,
    )


def test_align_from_params() -> None:
    example_scan, _ = read_image(TEST_IMAGE_PATH)
    torch_scan = prepare_scan(example_scan)
    model = load_alignment_model()

    # Compare the two alignment functions
    aligned, params = align_to_bean(torch_scan, model)
    aligned_from_params = transform_from_params(
        torch_scan, rotation=params["rotation"], translation=params["translation"], scaling=params["scaling"]
    )
    assert torch.all(aligned == aligned_from_params)

    # set certain parameters to default values
    aligned_from_params = transform_from_params(
        torch_scan, translation=params["translation"], scaling=params["scaling"]
    )
    aligned_from_params = transform_from_params(torch_scan, rotation=params["rotation"], scaling=params["scaling"])
    aligned_from_params = transform_from_params(
        torch_scan, rotation=params["rotation"], translation=params["translation"]
    )


def test_unalign_scan() -> None:
    """This test tests that a scan can be aligned and then unaligned to the original image,
    resulting in the same image."""
    example_scan, spacing = read_image(TEST_IMAGE_PATH)
    torch_scan = prepare_scan(example_scan)
    model = load_alignment_model()

    aligned_image, _, transform = align_to_bean(torch_scan, model, return_affine=True, scale=False)
    unaligned_im = transform_from_affine(aligned_image, transform.inverse())

    # write images to compare them manually, some interpolation artefacts are introduced
    # so difficult to compare max pixel values
    # write_image(
    #     TEMP_SAVE_PATH / "original.nii.gz",
    #     torch_scan.squeeze().cpu().numpy(),
    #     spacing=spacing,
    # )
    # write_image(
    #     TEMP_SAVE_PATH / 'unaligned/nii.gz',
    #     unaligned_im.squeeze().cpu().numpy(),
    #     spacing=spacing,
    # )

    # also write them to png to manually inspect
    fig_original = plot_midplanes(torch_scan.squeeze().cpu().numpy(), "Original")
    fig_unaligned = plot_midplanes(unaligned_im.squeeze().cpu().numpy(), "Unaligned")
    fig_original.savefig(TEMP_SAVE_PATH / 'original.png')
    fig_unaligned.savefig(TEMP_SAVE_PATH / "unaligned.png")


def test_scaling_twosteps() -> None:
    """This function test whether applying first alignment without scaling, and then applying the scaling seperately
    gives the same result as applying the alignment + scaling in one step."""
    example_scan, spacing = read_image(TEST_IMAGE_PATH)
    torch_scan = prepare_scan(example_scan)
    model = load_alignment_model()

    # 1 step approach
    aligned_scan, params = align_to_bean(torch_scan, model)

    # 2 step approach
    aligned_noscale, _ = align_to_bean(torch_scan, model, scale=False)
    aligned_twostep = transform_from_params(aligned_noscale, scaling=params["scaling"])

    # write images to compare them manually
    # write_image(
    #     Path( TEMP_SAVE_PATH / "aligned_scan_onestep.nii.gz"),
    #     aligned_scan.squeeze().cpu().numpy(),
    #     spacing=spacing,
    # )
    # write_image(
    #     Path( TEMP_SAVE_PATH / "aligned_scan_twostep.nii.gz"),
    #     aligned_twostep.squeeze().cpu().numpy(),
    #     spacing=spacing,
    # )

    # compare the two images
    max_diff = torch.max(torch.abs(aligned_scan - aligned_twostep))
    print(f"Max difference between the two images: {max_diff:.3f} on pixel range 0-1")

    # this threshold is quite arbitrary, better to check the similarities of the image visually in itk-snap
    assert max_diff < 0.25, "The two images are too different"


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
    aligned_scan_perm, _ = align_to_bean(torch_scan.permute(0, 1, 4, 3, 2), model)

    # odd no. axis permutation flips axis, so we have to correct for that in the results
    aligned_np = aligned_scan.squeeze().cpu().numpy()
    aligned_perm_np = aligned_scan_perm.squeeze().cpu().numpy()[::-1]

    # plot and save, these should look roughly similar (except for differences in stochasticity in network)
    fig_original = plot_midplanes(aligned_np, title="aligned scan")
    fig_permuted = plot_midplanes(aligned_perm_np, title="aligned scan perm")
    fig_original.savefig(TEMP_SAVE_PATH / "aligned_original.png")
    fig_permuted.savefig(TEMP_SAVE_PATH / "aligned_permuted.png")


def test_align_to_atlas() -> None:
    example_scan, spacing = read_image(TEST_IMAGE_PATH)
    torch_scan = prepare_scan(example_scan)
    model = load_alignment_model()

    # align scan (has to be scale = true to make transform to atlasspace work)
    aligned_scan, params, affine = align_to_bean(torch_scan, model, return_affine=True, scale=True)

    # get the transformation to atlas space
    atlas_transform = _get_transform_to_atlasspace()

    # 1 and 2 step approach
    aligned_atlas = apply_affine(aligned_scan, atlas_transform)
    aligned_atlas_direct = apply_affine(torch_scan, atlas_transform @ affine)

    assert isinstance(aligned_atlas, torch.Tensor)
    assert isinstance(aligned_atlas_direct, torch.Tensor)

    fig_2step = plot_midplanes(aligned_atlas.squeeze().cpu().numpy(), title="aligned atlas")
    fig_direct = plot_midplanes(aligned_atlas_direct.squeeze().cpu().numpy(), title="aligned atlas direct")

    fig_2step.savefig(TEMP_SAVE_PATH / "align_to_atlas_2step.png")
    fig_direct.savefig(TEMP_SAVE_PATH / "align_to_atlas_1step.png")


def test_align_to_atlas_direct() -> None:
    example_scan, spacing = read_image(TEST_IMAGE_PATH)
    torch_scan = prepare_scan(example_scan)
    model = load_alignment_model()

    aligned_to_atlas_unscaled, transform_dict = align_to_atlas(torch_scan, model, scale=False, return_affine=False)
    aligned_to_atlas_scaled, transform_dict = align_to_atlas(torch_scan, model, scale=True, return_affine=False)

    plot_midplanes(aligned_to_atlas_unscaled.squeeze().cpu().numpy(), title="unscaled")
    plot_midplanes(aligned_to_atlas_scaled.squeeze().cpu().numpy(), title="scaled")

    aligned_to_atlas_scaled_2step = transform_from_params(aligned_to_atlas_unscaled, scaling=transform_dict["scaling"])
    plot_midplanes(aligned_to_atlas_scaled_2step.squeeze().cpu().numpy(), title="unscaled")


def test_get_atlastransform() -> None:
    """test function to assert the generated atlas transformations is correct"""
    atlas_transform = _get_transform_to_atlasspace()

    assert atlas_transform.shape == (1, 4, 4)

    translation = atlas_transform[0, :3, 3]
    assert torch.allclose(translation, torch.tensor([7.8464, -0.1925, 8.5786]), atol=1e-4)

    rotation = atlas_transform[0, :3, :3]
    assert torch.allclose(
        rotation,
        torch.tensor([[-0.0033, 0.9995, -0.0320], [0.9997, 0.0025, -0.0255], [0.0254, 0.0321, 0.9992]]),
        atol=1e-4,
    )


def test_get_transform_to_atlasspace() -> None:
    full_atlasspace_transform = _get_transform_to_atlasspace()
    assert full_atlasspace_transform.shape == (1, 4, 4)

    translation = full_atlasspace_transform[0, :3, 3]
    assert torch.allclose(translation, torch.tensor([7.8464, -0.1925, 8.5786]), atol=1e-4)

    rotation = full_atlasspace_transform[0, :3, :3]
    assert torch.allclose(
        rotation,
        torch.tensor([[-0.0033, 0.9995, -0.0320], [0.9997, 0.0025, -0.0255], [0.0254, 0.0321, 0.9992]]),
        atol=1e-4,
    )
