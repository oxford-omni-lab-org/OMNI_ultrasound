import numpy as np
from pathlib import Path
import SimpleITK as sitk
from typing import Union
import nibabel as nib
from typeguard import typechecked
import matplotlib.pyplot as plt


@typechecked
def _read_mha_image(
    vol_path: Path, return_info: bool = False
) -> Union[np.ndarray, tuple[np.ndarray, tuple[float, float, float]]]:
    """reads an sitk image, and returns the numpy array and optionally the spacing

    :param vol_path: path to read the image from
    :param return_info: whether to return the spacing and origin, defaults to False
    :return: np.ndarray of the image or tuple of (np.ndarray, spacing)
    """

    assert vol_path.exists(), f"vol_path does not exist: {vol_path}"

    itk_img = sitk.ReadImage(str(vol_path))
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
    """reads an nifti image, and returns the numpy array and optionally the spacing
    :param vol_path: path to read the image from
    :param return_info: whether to return the spacing and origin, defaults to False
    :return: np.ndarray of the image or tuple of (np.ndarray, spacing)
    """
    assert vol_path.exists(), f"vol_path does not exist: {vol_path}"

    nib_img: nib.Nifti1Image = nib.nifti1.load(vol_path)

    # extract the data
    im_array = nib_img.get_fdata()
    if return_info:
        spacing = nib_img.header.get_zooms()
        spacing = tuple([float(s) for s in spacing])
        return im_array, spacing
    else:
        return im_array


@typechecked
def read_image(vol_path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """reads in an image, and returns the numpy array and spacing

    :param vol_path: path to the image
    :raises ValueError: when the path has an unknown file extension
    :return: the numpy array and spacing
    """
    if vol_path.suffix == ".mha":
        nii_array, spacing = _read_mha_image(vol_path, return_info=True)

    elif vol_path.suffix in [".nii", ".gz"]:
        nii_array, spacing = _read_nii_image(vol_path, return_info=True)

    else:
        raise ValueError(f"Unknown file extension: {vol_path.suffix}")

    return nii_array, spacing


def generate_nii_affine(spacing: tuple[float, float, float]) -> np.ndarray:
    """generate the affine matrix from the spacing.
    The negative spacing for the first two dimensions is to ensure consistency with the mha format.

    :param spacing: the spacing of the image
    :return: the affine matrix
    """

    affine = np.eye(4, 4)
    affine[0, 0] = -spacing[0]
    affine[1, 1] = -spacing[1]
    affine[2, 2] = spacing[2]
    return affine


@typechecked
def write_image(vol_path: Path, image_array: np.ndarray, spacing: tuple[float, float, float]) -> None:
    """Saves a numpy array as an nifti or mha image, the type is determined by the file extension

    :param vol_path: path to save the images to
    :param image_array: numpy array with the image data
    :param spacing: spacing of the pixels, defaults to (0.6, 0.6, 0.6)
    """

    # check inputs
    assert 2 <= len(image_array.shape) <= 5, "im_array must be 2D-4D"
    assert len(spacing) == len(image_array.shape), "spacing must have the same no. dimensions as the image array"

    # makes directory for saving if it doesn't exist
    if not vol_path.parent.exists():
        vol_path.parent.mkdir(parents=True)

    if vol_path.suffix == ".mha":
        assert len(image_array.shape) == 3, "mha images only supported for 3D images"
        sitk_array = np.transpose(image_array, (2, 1, 0)).astype("int16")
        sitk_im = sitk.GetImageFromArray(sitk_array)

        # set the spacing
        sitk_im.SetSpacing(spacing)
        sitk_im.SetOrigin((0, 0, 0))

        # save the image
        sitk.WriteImage(sitk_im, str(vol_path))

    elif vol_path.suffix in [".nii", ".gz"]:
        # we are assuming nifti 2 here
        affine = generate_nii_affine(spacing)
        im = nib.Nifti1Image(image_array, affine=affine.astype("float64"), dtype=np.int16)

        # save the image
        nib.save(im, str(vol_path))

    else:
        raise ValueError(f"Unknown file extension: {vol_path.suffix}")


def plot_midplanes(image: np.ndarray, title: str) -> plt.Figure:
    """plot the midplanes of a 3D image

    :param image: 3D array of image data
    :param title: title of the plot
    :return: reference to the plot
    """

    assert len(image.shape) == 3, "image must be 3D"

    midplanes = np.array(image.shape) // 2
    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(image[midplanes[0], :, :], cmap="gray")
    axes[0].set_axis_off()
    axes[0].set_title("1 plane")

    axes[1].imshow(image[:, midplanes[1], :], cmap="gray")
    axes[1].set_axis_off()
    axes[1].set_title("2 plane")

    axes[2].imshow(image[:, :, midplanes[2]], cmap="gray")
    axes[2].set_axis_off()
    axes[2].set_title("3 plane")

    fig.suptitle(title)
    fig.tight_layout()

    return fig
