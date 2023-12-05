from pathlib import Path
import numpy as np
import pytest
import doctest
from fetalbrain.utils import _read_mha_image, _read_nii_image, read_image, write_image, plot_midplanes


def test_write_mha_image() -> None:
    """test write image function"""
    image = np.random.rand(10, 10, 10)
    # write as a .mha
    savepath = Path('test.mha')
    write_image(savepath, image, spacing=(0.6, 0.6, 0.6))
    assert savepath.exists()