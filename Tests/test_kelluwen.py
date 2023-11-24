import sys
import torch


sys.path.append("/home/sedm6226/Documents/Projects/US_analysis_package")

from src.alignment.kelluwen_transforms import (  # noqa: E402
    deconstruct_affine,
    generate_affine,
    apply_affine,
    generate_rotation,
    generate_scaling,
    generate_translation,
)


def test_deconstruct_affine() -> None:
    """test deconstruct_affine function"""

    transform_affine = torch.eye(4, 4, dtype=torch.float32).unsqueeze(0)
    transl, rot, scale = deconstruct_affine(
        transform_affine, transform_order="srt", type_rotation="euler_xyz", type_output="positional"
    )

    assert isinstance(transl, torch.Tensor)
    assert isinstance(rot, torch.Tensor)
    assert isinstance(scale, torch.Tensor)
    assert transl.shape == (1, 3)
    assert rot.shape == (1, 3)
    assert scale.shape == (1, 3)

    transform_affine = transform_affine.unsqueeze(0)
    transl, rot, scale = deconstruct_affine(
        transform_affine, transform_order="srt", type_rotation="euler_xyz", type_output="positional"
    )

    assert isinstance(transl, torch.Tensor)
    assert isinstance(rot, torch.Tensor)
    assert isinstance(scale, torch.Tensor)
    assert transl.shape == (1, 1, 3)
    assert rot.shape == (1, 1, 3)
    assert scale.shape == (1, 1, 3)


def test_apply_affine() -> None:
    image = torch.rand((1, 1, 160, 160, 160))
    identity_affine = torch.eye(4, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # for origin = 'origin' the image should not change
    image_transformed = apply_affine(image, identity_affine, type_origin="origin")
    assert isinstance(image_transformed, torch.Tensor)
    assert torch.allclose(image_transformed, image, atol=1e-9)

    # for origin = 'centre' there is a very small shift caused by the rounding errors
    # when computing the centre_transform and centre_transform.inverse()


def test_generate_affine() -> None:
    parameter_translation = torch.rand((1, 3))
    parameter_rotation = torch.rand((1, 4))
    parameter_scaling = torch.rand((1, 3))

    transform_affine = generate_affine(
        parameter_translation, parameter_rotation, parameter_scaling, type_rotation="quaternions"
    )

    assert isinstance(transform_affine, torch.Tensor)
    assert transform_affine.shape == (1, 4, 4)


def test_generate_translation() -> None:
    parameter_translation = torch.rand((1, 3))
    transform_translation = generate_translation(parameter_translation)

    assert isinstance(transform_translation, torch.Tensor)
    assert transform_translation.shape == (1, 4, 4)


def test_generate_scaling() -> None:
    parameter_scaling = torch.rand((1, 3))
    transform_scaling = generate_scaling(parameter_scaling)

    assert isinstance(transform_scaling, torch.Tensor)
    assert transform_scaling.shape == (1, 4, 4)


def test_generate_rotation() -> None:
    parameter_rotation = torch.rand((1, 4))
    transform_rotation = generate_rotation(parameter_rotation, type_rotation="quaternions")
    assert isinstance(transform_rotation, torch.Tensor)
    assert transform_rotation.shape == (1, 4, 4)

    parameter_rotation = torch.rand((1, 3))
    transform_rotation = generate_rotation(parameter_rotation, type_rotation="euler_xyz")
    assert isinstance(transform_rotation, torch.Tensor)
    assert transform_rotation.shape == (1, 4, 4)
