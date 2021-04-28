from file_utils import read_rgb_image, read_depth_image
from point_cloud import depth_to_xyz, xyz_to_volume, clip_by_border
from model_functions import display_rgbd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def main():
    rgb = read_rgb_image("./data/000000.png")
    intel = (69.4, 42.5)
    depth_empty = read_depth_image("./data/000000.raw", shape=(480, 640))
    depth_full = read_depth_image("./data/000004.raw", shape=(480, 640))

    depth_empty = depth_empty.astype("float32") / 1000.
    depth_full = depth_full.astype("float32") / 1000.

    xyz_empty = depth_to_xyz(depth_empty, fov=intel)
    xyz_full = depth_to_xyz(depth_full, fov=intel)

    xyz_empty = clip_by_border(xyz_empty, (-1., 1.), (-1., 1.), (2.4, 1.))
    xyz_full = clip_by_border(xyz_full, (-1., 1.), (-1., 1.), (2.4, 1.))

    # xyz_empty = xyz_empty[np.random.choice(xyz_empty.shape[0], 100), :]
    # xyz_full = xyz_full[np.random.choice(xyz_full.shape[0], 100), :]

    xyz_empty[:, 2] = 1.4 - xyz_empty[:, 2]
    xyz_full[:, 2] = 1.4 - xyz_full[:, 2]

    vol_empty = xyz_to_volume(xyz_empty)
    vol_full = xyz_to_volume(xyz_full)

    print("Volume of full container= ", vol_full, "m^3")
    print("Volume of empty container=", vol_empty, "m^3")
    print("Difference in volume=     ", vol_full - vol_empty, "m^3")

    # display_rgbd([rgb, depth_empty])
    display_point_cloud(np.concatenate((xyz_full, xyz_empty), axis=0))
    return None


def display_point_cloud(xyz):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter3D(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=xyz[:, 2], cmap='hsv')
    ax.set_title("3D plot")
    plt.show()
    return None


if __name__ == '__main__':
    main()
