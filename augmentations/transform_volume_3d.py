import random
import torch

from torch import nn
from typing import Tuple, List, Optional, Sequence, Union
import numpy as np
from monai.transforms.utils import map_spatial_axes, ensure_tuple
# import open3d as o3d


class Compose:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Flip(nn.Module):
    """
    Reverses the order of elements along the given spatial axis. Preserves shape.
    Uses ``np.flip`` in practice. See numpy.flip for additional details:
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html.
    Args:
        spatial_axis: spatial axes along which to flip over. Default is None.
            The default `axis=None` will flip over all of the axes of the input array.
            Other accepted values are 0, 1 and 2
    """

    def __init__(self, spatial_axis: Optional[int] = None) -> None:
        super(Flip, self).__init__()
        self.spatial_axis = spatial_axis
        assert spatial_axis in [None, 0, 1, 2], "Unaccepted value provided for the spatial axis"

    def forward(self, img: np.ndarray, pc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]),
        """
        result: np.ndarray = np.flip(img, map_spatial_axes(img.ndim, self.spatial_axis))
        flipped_pc: np.ndarray = self.flip_pc_using_axes_coordinate(points=pc)
        return result.astype(img.dtype), flipped_pc

    def flip_pc_using_axes_coordinate(self, points):
        """
        We assume that the Z axis is pointing upwards.
        :param point_cloud: Original points to be rotated
        :return:
        """
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        rotation_angle = None

        if self.spatial_axis == 2:
            points[:, [1, 2]] = 1 - points[:, [1, 2]]
            # Flip z is equivalent to rotation along x-axis
            # rotation_angle = point_cloud.get_rotation_matrix_from_axis_angle((np.pi / 2, 0, 0))
        elif self.spatial_axis == 0:
            # flip x is rotate along Y
            # rotation_angle = point_cloud.get_rotation_matrix_from_axis_angle((0, -np.pi / 2, 0))
            points[:, 0] = 1 - points[:, 0]
        elif self.spatial_axis == 1:
            # flip y is rotation along z-axis
            # rotation_angle = point_cloud.get_rotation_matrix_from_axis_angle((0, 0, np.pi / 2))
            points[:, [0, 1]] = 1 - points[:, [0, 1]]
        else:
            # The spatial axis is None in this case
            points = 1 - points
        if rotation_angle is not None:
            return np.asarray(point_cloud.rotate(rotation_angle).points)
        return points


class Rotate90(nn.Module):
    """
    Rotate an array by 90 degrees in the plane specified by `axes`.
    See np.rot90 for additional details:
    https://numpy.org/doc/stable/reference/generated/numpy.rot90.html.
    """

    def __init__(self, k: int = 1, spatial_axes: Tuple[int, int] = (0, 1)) -> None:
        """
        Args:
            k: number of times to rotate by 90 degrees.
            spatial_axes: 2 int numbers, defines the plane to rotate with 2 spatial axes.
                Default: (0, 1), this is the first two axis in spatial dimensions.
                Available choices are (0, 1), (0, 2) and (1, 2).
        """
        super(Rotate90, self).__init__()
        self.k = k
        spatial_axes_: Tuple[int, int] = ensure_tuple(spatial_axes)  # type: ignore
        if len(spatial_axes_) != 2:
            raise ValueError("spatial_axes must be 2 int numbers to indicate the axes to rotate 90 degrees.")
        assert spatial_axes_ in [(0, 1), (0, 2), (1, 2)]
        self.spatial_axes = spatial_axes_

    def __call__(self, img: np.ndarray, pc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]),
        """
        #

        result: np.ndarray = np.rot90(img, self.k, map_spatial_axes(img.ndim, self.spatial_axes))
        rotated_points = self.rotate_pc_using_axes_coordinate(points=pc)
        return result.astype(img.dtype), rotated_points

    def rotate_pc_using_axes_coordinate(self, points):
        """
        We assume that the Z axis is pointing upwards.
        :param point_cloud: Original points to be rotated
        :return:
        """
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)

        if self.spatial_axes == (0, 1):
            # This is the X-Y plane, thus we rotate about the z-axis
            rotation_angle = point_cloud.get_rotation_matrix_from_axis_angle((0, 0, np.pi / 2))
        elif self.spatial_axes == (1, 2):
            rotation_angle = point_cloud.get_rotation_matrix_from_axis_angle((np.pi / 2, 0, 0))
        else:
            rotation_angle = point_cloud.get_rotation_matrix_from_axis_angle((0, -np.pi / 2, 0))
        return np.asarray(point_cloud.rotate(rotation_angle, center=(0.5, 0.5, 0.5)).points)


class RandomTransforms:
    """Base class for a list of transformations with randomness
    Args:
        transforms (sequence): list of transformations
    """

    def __init__(self, transforms):
        if not isinstance(transforms, Sequence):
            raise TypeError("Argument transforms should be a sequence")
        self.transforms = transforms

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomlySelectTransform:
    def __init__(self, *args):
        self.transforms = args

    def __call__(self, img, pc, augmentations):
        for aug in augmentations:
            img, pc = aug(img, pc)
        return img, pc

    def get_augmentation(self):
        return [random.choice(self.transforms)]


class RandomOrder:
    """Apply a list of transformations in a random order. This transform does not support torchscript.
    """

    def __init__(self, *args):
        self.transforms = args

    def __call__(self, img, pc, order_for_augmentations):
        for i in order_for_augmentations:
            img, pc = self.transforms[i](img, pc)
        return img, pc

    def get_augmentation(self):
        order = list(range(len(self.transforms)))
        random.shuffle(order)
        return order


if __name__ == '__main__':
    x = np.random.randn(1, 8, 8, 8)
    pc = np.random.randn(8, 3)  # 8 vertices in total
    t = RandomOrder(Rotate90(spatial_axes=(0, 1)), Rotate90(spatial_axes=(0, 2)), Rotate90(spatial_axes=(1, 2)))
    # f = Flip()
    x_prime, pc_rot = t(x, pc, t.get_augmentation())
    # x_prime, pc_rot = f(x, pc)
    print(pc_rot.shape)
