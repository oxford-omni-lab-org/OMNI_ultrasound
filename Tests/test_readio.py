from pathlib import Path
import numpy as np
import pytest
import doctest
from fetalbrain.utils import _read_mha_image, _read_nii_image, read_image, write_image, plot_midplanes


TEST_IMAGE_PATH_MHA = Path("test_data/09-8515_187days_1049.mha")
TEST_IMAGE_PATH_INVALID = Path("test_data/09-8515_187days_1049.lb")
TEST_SAVEPATH_MHA = Path("test_data/new_09-8515_187days_1049.mha")
TEST_SAVEPATH_NII = Path("test_data/new_09-8515_187days_1049.nii.gz")

#doctest.testmod()


def test_read_sitk_image() -> None:
    """test read image function"""
    img_only = _read_mha_image(TEST_IMAGE_PATH_MHA)
    img, origin = _read_mha_image(TEST_IMAGE_PATH_MHA, return_info=True)

    assert len(img.shape) == len(origin)
    assert np.all(img_only == img)


def test_write_mha_image() -> None:
    """test write image function"""
    img, spacing = _read_mha_image(TEST_IMAGE_PATH_MHA, return_info=True)

    # write as a .mha
    write_image(TEST_SAVEPATH_MHA, img, spacing=spacing)
    assert TEST_SAVEPATH_MHA.exists()


def test_write_nii_image() -> None:
    """test write image function"""
    img, spacing = _read_mha_image(TEST_IMAGE_PATH_MHA, return_info=True)

    # write as a .nii
    write_image(TEST_SAVEPATH_NII, img, spacing=spacing)
    assert TEST_SAVEPATH_NII.exists()


def test_read_nii_image() -> None:
    """test read image function"""
    img_only = _read_nii_image(TEST_SAVEPATH_NII)
    img, spacing = _read_nii_image(TEST_SAVEPATH_NII, return_info=True)

    assert len(img.shape) == len(spacing)
    assert np.all(img_only == img)


def test_read_image() -> None:
    """test read image function for .nii and .mha"""

    # test for .mha image
    img_from_mha, spacing = read_image(TEST_IMAGE_PATH_MHA)
    assert len(img_from_mha.shape) == len(spacing)

    # test for .nii image
    img_from_nii, origin = read_image(TEST_SAVEPATH_NII)
    assert len(img_from_nii.shape) == len(origin)

    # we are working with float numbers so small tolerance
    assert np.all(img_from_nii - img_from_mha < 1e-7)

    # test for invalid path name
    with pytest.raises(ValueError):
        read_image(TEST_IMAGE_PATH_INVALID)


def test_readwrite_consistency() -> None:
    """test that writing and reading the image does not change propoerties"""
    img, spacing = read_image(TEST_IMAGE_PATH_MHA)

    # write and read as mha
    write_image(TEST_SAVEPATH_MHA, img, spacing=spacing)
    img_reload, spacing_reload = read_image(TEST_SAVEPATH_MHA)

    assert np.all(img == img_reload)
    assert np.all(spacing == spacing_reload)

    # write and read as nii
    write_image(TEST_SAVEPATH_NII, img, spacing=spacing)
    img_reload, spacing_reload = read_image(TEST_SAVEPATH_NII)

    assert np.all(img == img_reload)

    # we are working with float numbers so small tolerance
    assert np.all(img - img_reload < 1e-7)


def test_plot_midplanes() -> None:
    """test plot midplanes function"""
    img, _ = read_image(TEST_IMAGE_PATH_MHA)
    fig = plot_midplanes(img, "test")
    assert fig is not None
