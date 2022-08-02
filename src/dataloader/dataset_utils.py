import cv2
import numpy as np
from math import atan2, degrees, pi
import pandas as pd
from angles import normalize
import torch
import traceback
# pytorch3D imports
try:
    from pytorch3d.renderer import look_at_rotation
except ImportError:
    print('PyTorch3D could not be imported.')
    traceback.print_exc()

def shortest_angular_distance(theta1, theta2):
    """
    Calculates the shortest angular distance between two angles

    Args:
        theta1 (float): Angle in degrees
        theta2 (float): Angle in degrees

    Returns:
        float: Shortest angular distance between theta1 and theta2
    """
    dtheta = (theta2 - theta1) % 360
    dtheta = dtheta if dtheta <= 180 else dtheta - 360
    return abs(dtheta)


def calc_mu_sigma(dataset):
    """
    Example usage:

    train_transform = torchvision.transforms.Compose([Resize((224, 224))])
    train_dataset_path = os.path.abspath('../datasets/vkitti2/Scene01/')
    train_dataset = VirtualKITTIv2Dataset(
        root_dir=train_dataset_path,
        categories=["clone", "fog"],
        formats=["rgb", "rgb"],
        return_pairs=True,
        transform=train_transform
    )

    mean, sigma = calc_mu_sigma(train_dataset)
    """
    mean = np.zeros((2, 3))
    std = np.zeros((2, 3))
    n = len(dataset)

    for idx in range(n):
        img1, img2, _ = dataset[idx]
        img1 = cv2.normalize(img1, None, alpha=0.0, beta=1.0,
                             norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img2 = cv2.normalize(img2, None, alpha=0.0, beta=1.0,
                             norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        mean[0, :] += img1.mean(axis=0).mean(axis=0)
        mean[1, :] += img2.mean(axis=0).mean(axis=0)
        std[0, :] += img1.std(axis=0).std(axis=0)
        std[1, :] += img2.std(axis=0).std(axis=0)

    return mean / n, std / n


def normalize_angle(a):
    """
    Normalize an angle to range -180:180

    Args:
        a (float): The angle to normalize

    Returns:
        float: The normalized angle
    """
    a = a % 360
    if a > 180:
        a -= 360
    return a


def shortest_angular_distance2(from_angle, to_angle):
    """
    [summary]

    Args:
        a ([type]): [description]
    """
    return normalize(to_angle - from_angle)


def calc_theta_global(pose: pd.DataFrame, to_degrees: bool = False):
    ''' Calculates the global angle of an observed KITTI object

    Args:
        pose (pd.Dataframe): The pose dataframe of the object

    Details:
        The convention in vkitti2 is that the yaw of an object vehicle is 0 iff
        its orientation is perpendicular to the ego vehicle and facing in the
        same direction as the x axis of the ego vehicle. Here 90 degrees are
        added to the global angle of the object vehicle so that a 0 angle
        denotes that the object vehicle has the same orientation as the ego
        vehicle.

    Returns:
        float: The global angle
    '''
    # 0: obj-veh. right facing
    angle = pose['rotation_camera_space_y']
    # 0: obj-veh. same direction
    # angle = pose['rotation_camera_space_y'] + pi / 2

    return degrees(angle) if to_degrees else angle


def calc_theta_ray(pose: pd.DataFrame, to_degrees: bool = False):
    ''' Calculates the ray angle of an observed KITTI object

    Args:
        pose (pd.Dataframe): The pose dataframe of the object

    Returns:
        float: The ray angle
    '''
    angle = atan2(float(pose['camera_space_X']), float(pose['camera_space_Z']))

    return degrees(angle) if to_degrees else angle


def calc_theta_local(pose: pd.DataFrame, offset=0, to_degrees: bool = False):
    ''' Calculates the local angle of an observed KITTI object

    Args:
        pose (pd.Dataframe): The pose dataframe of the object

    Returns:
        float: The local angle
    '''
    theta_global = calc_theta_global(pose, to_degrees=to_degrees)  # global angle
    theta_ray = calc_theta_ray(pose, to_degrees=to_degrees)  # ray angle
    theta_local = theta_global - theta_ray + offset  # local angle

    # normalize local angle
    lower_bound = -180 if to_degrees else -pi
    upper_bound = 180 if to_degrees else pi
    theta_local = normalize(theta_local, lower_bound, upper_bound)

    return theta_local


def tensor2numpy(tensor):
    img = tensor.numpy().transpose(1,2,0)
    img = 255.0 * (img - img.min()) / (img.max() - img.min())
    img = img.astype(np.uint8)
    return img


def tensor2numpyFloat(tensor):
    img = tensor.detach().cpu().squeeze().numpy().transpose(1,2,0)
    img = (img - img.min()) / (img.max() - img.min())
    return img


def tensor2numpyUInt8(tensor):
    img = tensor.detach().cpu().squeeze().numpy().transpose(1,2,0)
    img = 255.0 * (img - img.min()) / (img.max() - img.min())
    return img.astype(np.uint8)


def rotation_theta(theta, device_=None):
    """ borrowed from
        https://github.com/gkouros/NeMo/blob/main/code/lib/MeshUtils.py
    """
    # cos -sin  0
    # sin  cos  0
    # 0    0    1
    if type(theta) == float:
        if device_ is None:
            device_ = 'cpu'
        theta = torch.ones((1, 1, 1)).to(device_) * theta
    else:
        if device_ is None:
            device_ = theta.device
        theta = theta.view(-1, 1, 1)

    mul_ = torch.Tensor([[1, 0, 0, 0, 1, 0, 0, 0, 0], [0, -1, 0, 1, 0, 0, 0, 0, 0]]).view(1, 2, 9).to(device_)
    bia_ = torch.Tensor([0] * 8 + [1]).view(1, 1, 9).to(device_)

    # [n, 1, 2]
    cos_sin = torch.cat((torch.cos(theta), torch.sin(theta)), dim=2).to(device_)

    # [n, 1, 2] @ [1, 2, 9] + [1, 1, 9] => [n, 1, 9] => [n, 3, 3]
    trans = torch.matmul(cos_sin, mul_) + bia_
    trans = trans.view(-1, 3, 3)

    return trans


def campos_to_R_T(campos, theta, device, at=((0, 0, 0),), up=((0, 1, 0),)):
    """ adapted from
        https://github.com/gkouros/NeMo/blob/main/code/lib/MeshUtils.py
    """
    R = look_at_rotation(campos, at=at, device=device, up=up)
    R = torch.bmm(R, rotation_theta(theta, device_=device))
    T = -torch.bmm(R.transpose(1, 2), campos.unsqueeze(2))[:, :, 0]
    return R, T


def quaternion_distance(q1, q2, return_degrees=False):
    assert q1.shape == (4,) or (len(q1.shape) == 2 and q1.shape[1] == 4)
    assert q2.shape == (4,) or (len(q2.shape) == 2 and q2.shape[1] == 4)
    if isinstance(q1, torch.Tensor) and isinstance(q2, torch.Tensor):
        q1 = q1.numpy()
        q2 = q2.numpy()
    prod = abs(np.clip(np.dot(q1, q2), -1, 1))
    dist = 2 * np.arccos(prod)
    return degrees(dist) if return_degrees else dist
