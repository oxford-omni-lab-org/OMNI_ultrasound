"""
This modules contains helper functions for read and write I/O functions. Please use the functions to read/write
images when using the packages in this repository to ensure consistency. For example, niibabel and simpleITK read
in image differently, which can cause problems when using these together. The read/write functions in this module
have all been adapated so that they are consistent with each other.

For reading .nii / .nii.gz / .mha use:
    >>> image, spacing = read_image(vol_path)

For reading matlab image files use:
    >>> image, spacing = read_matlab_image(vol_path)

For writing .nii / .nii.gz / .mha use:
    >>> write_image(vol_path, image, spacing)
"""

import numpy as np
from pathlib import Path
import SimpleITK as sitk
import warnings
from typing import Union
import nibabel as nib
from typing import Optional
from typeguard import typechecked
import matplotlib.pyplot as plt

TEST_IMAGE_PATH = Path("test_data/09-8515_187days_1049.mha")
TEST_SAVEPATH_MHA = Path("test_data/test.mha")


@typechecked
def _read_mha_image(
    vol_path: Path, return_info: bool = False
) -> Union[np.ndarray, tuple[np.ndarray, tuple[float, float, float]]]:
    """Reads an sitk image, and returns the numpy array and optionally the spacing

    Args:
        vol_path: path to read the image from
        return_info: whether to return the spacing and origin, defaults to False

    Returns:
        nii_array: np.ndarray of the image
        itk_img.GetSpacing() (optional): spacing of the image

    """

    assert vol_path.exists(), f"vol_path does not exist: {vol_path}"
    assert vol_path.suffix == ".mha", f"vol_path must be a .mha file: {vol_path}"

    itk_img = sitk.ReadImage(vol_path.absolute().as_posix())
    sitk_array = sitk.GetArrayFromImage(itk_img)

    # swap axes to match nii convention [z, y, x]
    assert len(sitk_array.shape) == 3, "mha images only supported for 3D images"
    nii_array = np.transpose(sitk_array, (2, 1, 0))

    if return_info:
        return nii_array, itk_img.GetSpacing()
    else:
        return nii_array


@typechecked
def _read_nii_image(
    vol_path: Path, return_info: bool = False
) -> Union[np.ndarray, tuple[np.ndarray, tuple[float, float, float]]]:
    """Reads an nifti image, and returns the numpy array and optionally the spacing

    Args:
        vol_path: path to read the image from
        return_info: whether to return the spacing and origin, defaults to False

    Returns:
        im_array: np.ndarray of the image of size [H, W, D]
        spacing: spacing of the image (optional)
    """
    assert vol_path.exists(), f"vol_path does not exist: {vol_path}"
    assert vol_path.suffix in [".nii", ".gz"], f"vol_path must be a .nii or .nii.gz file: {vol_path}"

    nib_img: nib.Nifti1Image = nib.nifti1.load(vol_path)

    # extract the data
    im_array = np.asarray(nib_img.dataobj)
    if return_info:
        spacing = nib_img.header.get_zooms()
        spacing = tuple([float(s) for s in spacing])
        return im_array, spacing
    else:
        return im_array


@typechecked
def read_image(vol_path: Union[Path, str]) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Reads in an image, and returns the numpy array and spacing

    Args:
        vol_path: path to the image

    Raises:
        ValueError: when the path has an unknown file extension

    Returns:
        nii_array: np.ndarray of the image
        spacing: spacing of the image

    Example:
        >>> example_scan, spacing = read_image(TEST_IMAGE_PATH)
    """
    if isinstance(vol_path, str):
        vol_path = Path(vol_path)

    if vol_path.suffix == ".mha":
        nii_array, spacing = _read_mha_image(vol_path, return_info=True)

    elif vol_path.suffix in [".nii", ".gz"]:
        nii_array, spacing = _read_nii_image(vol_path, return_info=True)

    else:
        raise ValueError(f"Unknown file extension: {vol_path.suffix}")

    return nii_array, spacing


def read_matlab_image(vol_path: Union[Path, str]) -> tuple[np.ndarray, tuple[float, float, float]]:
    """This function can be used to read the matlab image file from the longitudinal atlas. To use this
    function the package needs to be install with the [matlab] addition (or this can be manually installed
    with pip install scipy). As there is no spacing information in the matlab files, this is hardcode
    to (0.6, 0.6, 0.6).

    Args:
        vol_path: path to the matlab image

    Returns:
        image_data: np.ndarray of the image
        spacing: spacing of the image (0.6, 0.6, 0.6)
    """
    from scipy.io import loadmat

    mat_img = loadmat(vol_path)
    img_data = mat_img["img_brain"][:, :, :, 0]

    # hardcoded spacing for matlab files
    spacing = (0.6, 0.6, 0.6)

    return img_data, spacing


def generate_nii_affine(spacing: tuple[float, float, float]) -> np.ndarray:
    """Generate the affine matrix from the spacing.
    The negative spacing for the first two dimensions is to ensure consistency with the mha format.

    Args:
        spacing: the spacing of the image

    Returns:
        affine: the affine matrix of size [4,4]

    Example:
        >>> affine = generate_nii_affine(spacing=(0.6, 0.6, 0.6))
        >>> assert affine.shape == (4,4)
    """

    affine = np.eye(4, 4)
    affine[0, 0] = -spacing[0]
    affine[1, 1] = -spacing[1]
    affine[2, 2] = spacing[2]
    return affine


def write_image(
    vol_path: Union[Path, str],
    image_array: np.ndarray,
    spacing: tuple[float, float, float] = (0.6, 0.6, 0.6),
    dtype: np.dtype = np.uint8,
    segm: bool = False,
) -> None:
    """Saves a numpy array as an nifti or mha image, the type is determined by the file extension.
    The images are written with integer pixel values between 0 and 255

    Args:
        vol_path: path to save the images to
        image_array: numpy array with the image data
        spacing: spacing of the pixels, defaults to (0.6, 0.6, 0.6)
        dtype: dtype of the saved image, defaults to np.uint8

    Example:
        >>> test_image =  np.random.rand(160, 160, 160)
        >>> write_image(TEST_SAVEPATH_MHA, test_image)
    """

    if isinstance(vol_path, str):
        vol_path = Path(vol_path)

    if np.max(image_array) <= 1 and not segm:
        warnings.warn(
            "Image array has values between 0 and 1, expects pixel values between 0 and 255\
                      for image data",
            UserWarning,
        )

    # check inputs
    assert 2 <= len(image_array.shape) <= 5, "im_array must be 2D-4D"
    assert len(spacing) == len(image_array.shape), "spacing must have the same no. dimensions as the image array"

    # makes directory for saving if it doesn't exist
    if not vol_path.parent.exists():
        vol_path.parent.mkdir(parents=True)

    if vol_path.suffix == ".mha":
        assert len(image_array.shape) == 3, "mha images only supported for 3D images"
        sitk_array = np.transpose(image_array, (2, 1, 0)).astype(dtype)
        sitk_im = sitk.GetImageFromArray(sitk_array)

        # set the spacing
        sitk_im.SetSpacing(spacing)
        sitk_im.SetOrigin((0, 0, 0))

        # save the image
        sitk.WriteImage(sitk_im, str(vol_path), useCompression=True)

    elif vol_path.suffix in [".nii", ".gz"]:
        # we are assuming nifti1 here
        affine = generate_nii_affine(spacing)
        im = nib.Nifti1Image(image_array.astype(dtype), affine=affine.astype("float64"))
        im.set_data_dtype(dtype)

        # save the image
        nib.save(im, str(vol_path))

    else:
        raise ValueError(f"Unknown file extension: {vol_path.suffix}")


def plot_midplanes(image: np.ndarray, title: str) -> plt.Figure:
    """Plot the midplanes of a 3D image

    Args:
        image: 3D array of image data
        title: title of the plot

    Returns:
        fig: reference to the plot

    Example:
        >>> example_scan, spacing = read_image(TEST_IMAGE_PATH)
        >>> fig = plot_midplanes(example_scan, "Original")
    """

    assert len(image.shape) == 3, "image must be 3D"

    midplanes = np.array(image.shape) // 2
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    axes[0].imshow(image[midplanes[0], :, :], cmap="gray")
    axes[0].set_axis_off()
    axes[0].set_title("1 plane")

    axes[1].imshow(image[:, midplanes[1], :], cmap="gray")
    axes[1].set_axis_off()
    axes[1].set_title("2 plane")

    axes[2].imshow(image[:, :, midplanes[2]], cmap="gray")
    axes[2].set_axis_off()
    axes[2].set_title("3 plane")

    fig.suptitle(title, fontsize=30)
    fig.tight_layout()

    return fig


def plot_planes_segm(image: np.ndarray, segmentation: np.ndarray, title: Optional[str] = None) -> plt.Figure:
    """Plot planes of a 3D image and segmentation to demonstrate segmentation results.

    TO-DO: Plot this properly with a colorbar and labels (taking a dictionary as input with class names)

    Args:
        image: 3D array of image data
        segmentation: 3D array of segmentation data
        title: title of the plot

    Returns:
        fig: reference to the plot

    Example:
        >>> example_scan, spacing = read_image(TEST_IMAGE_PATH)
        >>> fig = plot_midplanes(example_scan, "Original")
    """

    assert len(image.shape) == 3, "image must be 3D"

    max_color = np.max(segmentation)

    midplanes = [70, 80, 90]
    midplanes = [80, 80, 80]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    axes[0].imshow(image[midplanes[0], :, :], cmap="gray")
    axes[0].imshow(
        segmentation[midplanes[0], :, :],
        alpha=0.5 * (segmentation[midplanes[0]] > 0),
        cmap="gist_rainbow",
        vmin=0,
        vmax=max_color,
    )
    axes[0].set_axis_off()
    axes[0].set_title("1 plane")

    axes[1].imshow(image[:, midplanes[1], :], cmap="gray")
    axes[1].imshow(
        segmentation[:, midplanes[1], :],
        alpha=0.5 * (segmentation[:, midplanes[1]] > 0),
        cmap="gist_rainbow",
        vmin=0,
        vmax=max_color,
    )
    axes[1].set_axis_off()
    axes[1].set_title("2 plane")

    axes[2].imshow(image[:, :, midplanes[2]], cmap="gray")
    axes[2].imshow(
        segmentation[:, :, midplanes[2]],
        alpha=0.5 * (segmentation[:, :, midplanes[2]] > 0),
        cmap="gist_rainbow",
        vmin=0,
        vmax=max_color,
    )
    axes[2].set_axis_off()
    axes[2].set_title("3 plane")

    if title is not None:
        fig.suptitle(title, fontsize=30)
    fig.tight_layout()

    return fig
