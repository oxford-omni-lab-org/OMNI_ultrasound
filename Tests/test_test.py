from pathlib import Path
import numpy as np
from fetalbrain.utils import write_image


def test_write_mha_image() -> None:
    """test write image function"""
    image = np.random.rand(10, 10, 10)
    # write as a .mha
    savepath = Path('test.mha')
    write_image(savepath, image, spacing=(0.6, 0.6, 0.6))
    assert savepath.exists()