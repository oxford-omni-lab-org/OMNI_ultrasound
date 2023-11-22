import torch
import json
import numpy as np
import matplotlib.pyplot as plt




def realign_volume(im_np: np.ndarray, direction: str = "FelipeToAtlas", type_im: str = "im") -> np.ndarray:
    # Prepare the volume for analysis
    scan = torch.from_numpy(im_np).float()

    params_to_atlas = json.load(open("src/alignment/config/25wks_Atlas(separateHems)_mean_warped.json", "r"))
    eu_param_to_atlas = torch.Tensor(params_to_atlas["eu_param"]).reshape(1, -1)
    tr_param_to_atlas = torch.Tensor(params_to_atlas["tr_param"]).reshape(1, -1)
    sc_param_to_atlas = torch.Tensor([1, 1, 1]).reshape(1, -1)

    # do transform
    if direction == "FelipeToAtlas":
        #scan = torch.flip(scan, [0])
        atlas_transform = torchparams_to_transform(eu_param_to_atlas, tr_param_to_atlas, sc_param_to_atlas).inverse()
    elif direction == "AtlasToFelipe":
        atlas_transform = torchparams_to_transform(eu_param_to_atlas, tr_param_to_atlas, sc_param_to_atlas)
        scan = scan.permute((1, 2, 0))

    # scan -= scan.min()
    # scan /= scan.max()
    scan = scan[None, None, :]

    if type_im == "im":
        scan_manually_aligned = transform_pytorch(scan * 255, atlas_transform, mode="bilinear")
    elif type_im == "segm":
        scan_manually_aligned = transform_pytorch(scan, atlas_transform, mode="nearest")

    # transform for saving
    if direction == "FelipeToAtlas":
        #transpose = (0, 1, 4, 2, 3)
        #scan_manually_aligned = scan_manually_aligned.permute(transpose)
        pass
    elif direction == "AtlasToFelipe":
        scan_manually_aligned = torch.flip(scan_manually_aligned, [2])
        # transpose = (0,1,2,4,3)
        # scan_manually_aligned = scan_manually_aligned.permute(transpose)

    numpy_aligned = np.squeeze(scan_manually_aligned.numpy()).astype(np.uint8)
    im = np.clip(numpy_aligned, 0, 255)

    return im


def torchparams_to_transform(euler_angles, shift_params, scale_params):
    """Converts parameters to pytorch transformation matric

    Args:
        euler_angles (list): [description]
        shift_params (list): [description]
        scale_params (int): [description]

    Returns:
        [type]: [description]
    """
    ea0 = torch.Tensor(euler_angles[:, 0])
    ea1 = euler_angles[:, 1]
    ea2 = euler_angles[:, 2]
    Id = torch.eye(4).reshape((1, 4, 4)).repeat(euler_angles.size()[0], 1, 1)

    Rx = Id.clone()
    Rx[:, 0, 0] = torch.cos(ea0)
    Rx[:, 0, 1] = torch.sin(ea0)
    Rx[:, 1, 0] = -torch.sin(ea0)
    Rx[:, 1, 1] = torch.cos(ea0)

    Ry = Id.clone()
    Ry[:, 0, 0] = torch.cos(ea1)
    Ry[:, 0, 2] = -torch.sin(ea1)
    Ry[:, 2, 0] = torch.sin(ea1)
    Ry[:, 2, 2] = torch.cos(ea1)

    Rz = Id.clone()
    Rz[:, 1, 1] = torch.cos(ea2)
    Rz[:, 1, 2] = torch.sin(ea2)
    Rz[:, 2, 1] = -torch.sin(ea2)
    Rz[:, 2, 2] = torch.cos(ea2)
    R = torch.matmul(Rx, torch.matmul(Ry, Rz))

    T = torch.eye(4).reshape((1, 4, 4)).repeat(shift_params.size()[0], 1, 1)
    T[:, 0, 3] = shift_params[:, 0]
    T[:, 1, 3] = shift_params[:, 1]
    T[:, 2, 3] = shift_params[:, 2]

    S = torch.eye(4).reshape((1, 4, 4)).repeat(scale_params.size()[0], 1, 1)
    S[:, 0, 0] = scale_params[:, 0]
    S[:, 1, 1] = scale_params[:, 1]
    S[:, 2, 2] = scale_params[:, 2]

    transform = torch.matmul(T, torch.matmul(R, S))
    return transform


# def matparams_to_torchparams(params_v1):
#     """Convert matlab parameters to parameters compatible with Torch

#     Args:
#         params_v1 (dict): dictionary with matlab parameters: keys are 'eu_param', 'tr_param', and 'sc_param'
#         can be loaded from ptah with scipy.io.loadmat(path)
#     """
#     # Adjust euler angles to work with pytorch
#     eu_param_v1_pytorch = -torch.flip(torch.Tensor(params_v1["eu_param"]).reshape(1, -1), [0, 1])

#     # Adjust translation parameters to work with pytorch. Note that for this step
#     # we're dividing by 80 due to the volume size used for alignment is 160x160x160
#     # and pytorch uses a normalized range of -1 to 1 (instead of -80 to 80).
#     tr_param_v1_pytorch = torch.flip(torch.Tensor(params_v1["tr_param"]).reshape(1, -1) / 80, [0, 1])

#     # Adjust scaling parameters to work with pytorch
#     sc_param_v1_pytorch = 1 / torch.Tensor(
#         [params_v1["sc_param"], params_v1["sc_param"], params_v1["sc_param"]]
#     ).reshape(1, -1)

#     eu_param_v1_pytorch = [float(f) for f in np.squeeze(eu_param_v1_pytorch.numpy())]
#     tr_param_v1_pytorch = [float(f) for f in np.squeeze(tr_param_v1_pytorch.numpy())]
#     sc_param_v1_pytorch = [float(f) for f in np.squeeze(sc_param_v1_pytorch.numpy())]

#     params_v1_pytorch = {
#         "eu_param": eu_param_v1_pytorch,
#         "tr_param": tr_param_v1_pytorch,
#         "sc_param": sc_param_v1_pytorch,
#     }

#     return params_v1_pytorch


def transform_pytorch(scan_tensor, transform, mode="bilinear"):
    """transforms a tensor with the given transformation matrix
    Args:
        scan_tensor (tensor): scan to be transformed
        transform (tensor): transformation matrix
        mode (str, optional): Interpolation method, use 'nearest' for binary masks. Defaults to 'bilinear'.

    Returns:
        tensor: transformed scan
    """
    grid = torch.nn.functional.affine_grid(
        transform[:, :3, :].to(scan_tensor.device),
        size=scan_tensor.shape,
        align_corners=False,
    ).to(scan_tensor.device)

    scan_manually_aligned = torch.nn.functional.grid_sample(scan_tensor, grid, align_corners=False, mode=mode)
    #    scan_manually_aligned = np.squeeze(scan_manually_aligned.numpy()).astype(
    #      np.uint8)

    return scan_manually_aligned




# def plot_midplanes_multi(image_list: list[np.ndarray], title_list: list[str]) -> plt.Figure:
#     """plot the midplanes of a 3D image

#     :param image: 3D array of image data
#     :param title: title of the plot
#     :return: reference to the plot
#     """

#     fig, axes = plt.subplots(len(image_list), 3)

#     for num in range(len(image_list)):
#         image = image_list[num]
#         title = title_list[num]

#         assert len(image.shape) == 3, "image must be 3D"

#         midplanes = np.array(image.shape) // 2


#         axes[num, 0].imshow(image[midplanes[0], :, :], cmap="gray")
#         axes[num, 0].set_axis_off()
#         axes[num, 0].set_title("1 plane")

#         axes[num, 1].imshow(image[:, midplanes[1], :], cmap="gray")
#         axes[num, 0].set_axis_off()
#         axes[num, 1].set_title("2 plane")

#         axes[num, 2].imshow(image[:, :, midplanes[2]], cmap="gray")
#         axes[num, 2].set_axis_off()
#         axes[num, 2].set_title("3 plane")

#     fig.suptitle(title)

#     return fig
