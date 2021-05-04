import numpy as np
from scipy.spatial import Delaunay, delaunay_plot_2d
import matplotlib.pyplot as plt
from math import pi


def depth_to_xyz(depth, fov=(82.1, 52.2)):
    """
    Takes a depth image array and calculates the points in xyz-space for all pixels

    :param depth: a depth image array (float [h, w])
    :param fov: the field of view angles in horizontal and vertical respectively (float, float)
    :return: an array of points in xyz-space (float [p, 3])
    """
    x_size = depth.shape[1]
    y_size = depth.shape[0]

    x = np.asarray([i - (x_size // 2) for i in range(x_size)])  # [w,]
    x = np.tile(np.expand_dims(x, axis=0), (y_size, 1))         # [h, w]
    x = np.tan(fov[0] * pi / 360) / (x_size / 2) * np.multiply(x, depth)

    y = np.asarray([i - (y_size // 2) for i in range(y_size)])  # [h,]
    y = np.tile(np.expand_dims(y, axis=-1), (1, x_size))        # [h, w]
    y = np.tan(fov[1] * pi / 360) / (y_size / 2) * np.multiply(y, depth)

    z = depth

    x = np.expand_dims(x, -1)  # [h, w, 1]
    y = np.expand_dims(y, -1)  # [h, w, 1]
    z = np.expand_dims(z, -1)  # [h, w, 1]
    print(x_size, y_size, x.shape, y.shape)
    p = np.concatenate((x, y, z), axis=-1)   # [h, w, 3]
    p = np.reshape(p, (x_size * y_size, 3))  # [p, 3]

    return p


def xyz_to_volume(xyz):
    """
    Calculates the volume of the points in xyz over the xy-plane.

    Negative z-coordinates result in negative volumes
    :param xyz: a list of coordinates in xyz-space that represent the point cloud (float [p, 3])
    :return: The total volume of the point cloud in xyz-space (float)
    """
    # Extract the triangles in the xy-plane
    triangles = Delaunay(xyz[:, 0:2])

    # Find all vertices of the constructed surface in xyz-space
    vertices = xyz[triangles.simplices]  # [e, points, xyz]
    x, y, z = vertices[:, :, 0], vertices[:, :, 1], vertices[:, :, 2]

    # Extract the mean height of all the triangular pillars
    heights = np.sum(z, axis=-1) / 3.  # [e,]
    # Extract the areas of all the triangles
    areas = np.abs(0.5 * (((x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0])) -
                          ((x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0]))))  # [e,]
    # Finding all volumes
    volumes = heights * areas  # [e,] = [e,] * [e,]
    volume = np.sum(volumes)   # [e,] -> [,]
    return volume


def bins_to_depth(depth_bins):
    """
    Converts a bin array into a depth image
    Copy of src.image_utils bins_to_depth, but without tensorflow dependencies

    :param depth_bins: the depth bins in one_hot encoding, shape (b, h, w, c),
    the depth bins can also be passed as softmax bins of shape (b, h, w, c)
    :return: a depth image of shape (b, h, w) with type tf.float32
    """
    bin_interval = (np.log10(80) - np.log10(0.25)) / 150
    # the borders variable here holds the depth for each specific value of the one hot encoded bins
    borders = np.asarray([np.log10(0.25) + (bin_interval * (i + 0.5)) for i in range(150)])
    depth = np.matmul(depth_bins, borders)  # [b, h, w, (c] * [c), 1] -> [b, h, w, 1]
    depth = np.power(10., depth)
    return depth


def clip_by_border(xyz, x_lim, y_lim, z_lim):
    xyz = xyz[np.where(xyz[:, 0] < max(x_lim))]
    xyz = xyz[np.where(xyz[:, 0] > min(x_lim))]
    xyz = xyz[np.where(xyz[:, 1] < max(y_lim))]
    xyz = xyz[np.where(xyz[:, 1] > min(y_lim))]
    xyz = xyz[np.where(xyz[:, 2] < max(z_lim))]
    xyz = xyz[np.where(xyz[:, 2] > min(z_lim))]
    return xyz
