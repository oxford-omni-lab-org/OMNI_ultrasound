import sys
from pathlib import Path
import torch

sys.path.append("/home/sedm6226/Documents/Projects/US_analysis_package")

from src.utils import read_image  # noqa: E402
from src.alignment.fBAN_v1 import AlignModel  # noqa: E402
from src.alignment.align import (  # noqa: E402
    load_alignment_model,
    unalign_scan,
    prepare_scan,
    align_scan,
    align_from_params,
)  # noqa: E402
from src.utils import write_image  # noqa: E402
from src.alignment.utils_alignment import plot_midplanes  # noqa: E402

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TEST_IMAGE_PATH = Path("src/alignment/test_data/06-5010_152days_0356.mha")


def test_load_alignment_model() -> None:
    model = load_alignment_model()
    assert isinstance(model, AlignModel)


def test_prepare_scan() -> None:
    example_scan, _ = read_image(TEST_IMAGE_PATH)
    torch_scan = prepare_scan(example_scan)
    assert torch_scan.shape == (1, 1, 160, 160, 160)


def test_align_scan() -> None:
    example_scan, spacing = read_image(TEST_IMAGE_PATH)
    torch_scan = prepare_scan(example_scan)
    model = load_alignment_model()

    aligned_scan, params = align_scan(torch_scan, model)

    # verify the scan is of the same shape as before
    assert aligned_scan.shape == torch_scan.shape

    assert "scaling" in params.keys()
    assert "rotation" in params.keys()
    assert "translation" in params.keys()

    assert params["scaling"].shape == (1, 3)
    assert params["rotation"].shape == (1, 4)
    assert params["translation"].shape == (1, 3)

    write_image(
        Path("src/alignment/test_data/aligned_scan.nii.gz"),
        aligned_scan.squeeze().cpu().numpy(),
        spacing=spacing,
    )


def test_align_from_params() -> None:
    example_scan, _ = read_image(TEST_IMAGE_PATH)
    torch_scan = prepare_scan(example_scan)
    model = load_alignment_model()

    # Compare the two alignment functions
    aligned, params = align_scan(torch_scan, model)
    aligned_from_params = align_from_params(
        torch_scan, rotation=params["rotation"], translation=params["translation"], scaling=params["scaling"]
    )
    assert torch.all(aligned == aligned_from_params)

    # set certain parameters to default values
    aligned_from_params = align_from_params(torch_scan, translation=params["translation"], scaling=params["scaling"])
    aligned_from_params = align_from_params(torch_scan, rotation=params["rotation"], scaling=params["scaling"])
    aligned_from_params = align_from_params(torch_scan, rotation=params["rotation"], translation=params["translation"])


def test_unalign_scan() -> None:
    """This test tests that a scan can be aligned and then unaligned to the original image,
    resulting in the same image."""
    example_scan, spacing = read_image(TEST_IMAGE_PATH)
    torch_scan = prepare_scan(example_scan)
    model = load_alignment_model()

    aligned_image, _, transform = align_scan(torch_scan, model, return_affine=True, scale=False)
    unaligned_im = unalign_scan(aligned_image, transform)

    # write images to compare them manually, some interpolation artefacts are introduced
    # so difficult to compare max pixel values
    write_image(
        Path("src/alignment/test_data/original.nii.gz"),
        torch_scan.squeeze().cpu().numpy(),
        spacing=spacing,
    )
    write_image(
        Path("src/alignment/test_data/unaligned.nii.gz"),
        unaligned_im.squeeze().cpu().numpy(),
        spacing=spacing,
    )

    # also write them to png to manually inspect
    fig_original = plot_midplanes(torch_scan.squeeze().cpu().numpy(), "Original")
    fig_unaligned = plot_midplanes(unaligned_im.squeeze().cpu().numpy(), "Unaligned")
    fig_original.savefig("src/alignment/test_data/original.png")
    fig_unaligned.savefig("src/alignment/test_data/unaligned.png")


def test_scaling_twosteps() -> None:
    """This function test whether applying first alignment without scaling, and then applying the scaling seperately
    gives the same result as applying the alignment + scaling in one step."""
    example_scan, spacing = read_image(TEST_IMAGE_PATH)
    torch_scan = prepare_scan(example_scan)
    model = load_alignment_model()

    # 1 step approach
    aligned_scan, params = align_scan(torch_scan, model)

    # 2 step approach
    aligned_noscale, _ = align_scan(torch_scan, model, scale=False)
    aligned_twostep = align_from_params(aligned_noscale, scaling=params["scaling"])

    # write images to compare them manually
    write_image(
        Path("src/alignment/test_data/aligned_scan_onestep.nii.gz"),
        aligned_scan.squeeze().cpu().numpy(),
        spacing=spacing,
    )
    write_image(
        Path("src/alignment/test_data/aligned_scan_twostep.nii.gz"),
        aligned_twostep.squeeze().cpu().numpy(),
        spacing=spacing,
    )

    # compare the two images
    max_diff = torch.max(torch.abs(aligned_scan - aligned_twostep))
    print(f"Max difference between the two images: {max_diff:.3f} on pixel range 0-1")

    # this threshold is quite arbitrary, better to check the similarities of the image visually in itk-snap
    assert max_diff < 0.15, "The two images are too different"


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
    aligned_scan, _ = align_scan(torch_scan, model)
    aligned_scan_perm, _ = align_scan(torch_scan.permute(0, 1, 4, 3, 2), model)

    # odd no. axis permutation flips axis, so we have to correct for that in the results
    aligned_np = aligned_scan.squeeze().cpu().numpy()
    aligned_perm_np = aligned_scan_perm.squeeze().cpu().numpy()[::-1]

    # plot and save, these should look roughly similar (except for differences in stochasticity in network)
    fig_original = plot_midplanes(aligned_np, title="aligned scan")
    fig_permuted = plot_midplanes(aligned_perm_np, title="aligned scan perm")
    fig_original.savefig("src/alignment/test_data/aligned_original.png")
    fig_permuted.savefig("src/alignment/test_data/aligned_permuted.png")
