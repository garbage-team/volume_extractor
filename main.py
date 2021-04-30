from file_utils import read_rgb_image, read_depth_image
from point_cloud import depth_to_xyz, xyz_to_volume, clip_by_border
from model_functions import display_rgbd
from run_model import run_model
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def main():
    intel = (69.4, 42.5)
    while True:
        rgb, depth = run_model()
        depth = depth[0]
        xyz = depth_to_xyz(depth, fov=intel)
        xyz_clipped = clip_by_border(xyz, (-.9, .9), (-.35, .35), (2.4, 1))
        xyz_clipped[:, 2] = 1.5 - xyz_clipped[:, 2]
        volume = xyz_to_volume(xyz_clipped)

        print("Volume in container: ", volume, "m^3")

        # display_rgbd([rgb, depth_empty])
        display_point_cloud(xyz_clipped)
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
