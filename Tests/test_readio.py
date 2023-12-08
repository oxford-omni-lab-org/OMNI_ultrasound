from pathlib import Path
import numpy as np
import pytest
from fetalbrain.utils import _read_mha_image, _read_nii_image, read_image, write_image, plot_midplanes
from path_literals import TEST_IMAGE_PATH, TEST_IMAGE_PATH_MHA, TEMP_SAVEPATH


def test_read_sitk_image() -> None:
    """test read image function"""
    img_only = _read_mha_image(TEST_IMAGE_PATH_MHA)
    img, origin = _read_mha_image(TEST_IMAGE_PATH_MHA, return_info=True)

    assert len(img.shape) == len(origin)
    assert np.all(img_only == img)


def test_write_mha_image() -> None:
    """test write image function"""
    img, spacing = _read_mha_image(TEST_IMAGE_PATH_MHA, return_info=True)

    # write as a .mha with default dtype
    temp_savepath_int = TEMP_SAVEPATH / TEST_IMAGE_PATH_MHA.name
    write_image(temp_savepath_int, img, spacing=spacing, dtype=np.uint8)
    assert temp_savepath_int.exists()

    # save as .mha with float datatype
    temp_savepath_flt = TEMP_SAVEPATH / (TEST_IMAGE_PATH_MHA.stem + "_float.mha")
    write_image(temp_savepath_flt, img, spacing=spacing, dtype=np.single)
    assert temp_savepath_flt.exists()

    # read the images back in to assert datatypes
    im_int, _ = read_image(temp_savepath_int)
    im_flt, _ = read_image(temp_savepath_flt)

    assert im_flt.dtype == np.single
    assert im_int.dtype == np.uint8


def test_read_nii_image() -> None:
    """test read image function"""
    img_only = _read_nii_image(TEST_IMAGE_PATH)
    img, spacing = _read_nii_image(TEST_IMAGE_PATH, return_info=True)

    assert len(img.shape) == len(spacing)
    assert np.all(img_only == img)


def test_write_nii_image() -> None:
    """test write image function"""
    img, spacing = _read_nii_image(TEST_IMAGE_PATH, return_info=True)

    # write as a .nii with default dtype
    temp_savepath_int = TEMP_SAVEPATH / TEST_IMAGE_PATH.name
    write_image(temp_savepath_int, img, spacing=spacing, dtype=np.uint8)
    assert temp_savepath_int.exists()

    # save as .nii with float datatype
    temp_savepath_flt = TEMP_SAVEPATH / (TEST_IMAGE_PATH.stem + "_float.nii.gz")
    write_image(temp_savepath_flt, img, spacing=spacing, dtype=np.float32)
    assert temp_savepath_flt.exists()

    # read the images back in to assert datatypes
    im_int, _ = read_image(temp_savepath_int)
    im_flt, _ = read_image(temp_savepath_flt)

    assert im_flt.dtype == np.float32
    assert im_int.dtype == np.uint8

    # assert that a warning is giving when writing with pixel values between 0 and 1
    temp_savepath_warn = TEMP_SAVEPATH / (TEST_IMAGE_PATH.stem + "_warn.nii.gz")
    with pytest.warns(UserWarning):
        write_image(temp_savepath_warn, img / 255, spacing=spacing)


def test_read_image() -> None:
    """test read image function for .nii and .mha"""

    # test for .mha image
    img_from_mha, spacing = read_image(TEST_IMAGE_PATH_MHA)
    assert len(img_from_mha.shape) == len(spacing)

    # test for .nii image
    img_from_nii, origin = read_image(TEST_IMAGE_PATH)
    assert len(img_from_nii.shape) == len(origin)

    # we are working with float numbers so small tolerance
    assert np.all(img_from_nii - img_from_mha < 1e-2)

    # test for invalid path name
    invalid_path = TEST_IMAGE_PATH.parent / 'invalid_path.kl'
    with pytest.raises(ValueError):
        read_image(invalid_path)


def test_readwrite_consistency() -> None:
    """test that writing and reading the image does not change propoerties"""
    img, spacing = read_image(TEST_IMAGE_PATH)

    # write and read as mha
    consis_path = TEMP_SAVEPATH / 'consistency.mha'
    write_image(consis_path , img, spacing=spacing, dtype=img.dtype)
    img_reload, spacing_reload = read_image(consis_path)

    assert np.all(img == img_reload)
    assert np.all(spacing == spacing_reload)

    # write and read as nii
    consis_path = TEMP_SAVEPATH / 'consistency.nii.gz'
    write_image(consis_path, img, spacing=spacing, dtype=img.dtype)
    img_reload, spacing_reload = read_image(consis_path)

    assert np.all(img == img_reload)


def test_plot_midplanes() -> None:
    """test plot midplanes function"""
    img, _ = read_image(TEST_IMAGE_PATH)
    fig = plot_midplanes(img, "test")
    assert fig is not None
