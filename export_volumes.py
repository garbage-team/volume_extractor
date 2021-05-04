import pathlib
import csv
from file_utils import read_depth_image
from point_cloud import depth_to_xyz, clip_by_border, xyz_to_volume


def export_volumes(input_path, fov=(69.4, 42.5)):
    """
    :param input_path: path to folder containing .raw depth images to extract volume of bins from
    :param fov: the field of view angles in horizontal and vertical respectively (float, float) (defualt: intel fov)
    :return: None
    """
    folder = pathlib.Path(input_path)
    depth_paths = list(folder.glob("*.raw"))
    volumes = []
    for path in depth_paths:
        print("Extracting volume from: ", path)
        depth_img = read_depth_image(path) / 1000
        xyz = depth_to_xyz(depth_img, fov=fov)
        xyz = clip_by_border(xyz, (-0.8, 0.8), (-0.35, 0.4), (2.4, 1.))
        xyz[:, 2] = 1.4 - xyz[:, 2]

        volume = xyz_to_volume(xyz)
        volumes.append([path, volume])
    print("Volumes extracted! Saving..")
    with open('volumes.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(['path', 'volume'])
        write.writerows(volumes)
    print("Volumes saved to: volumes.csv")
    print("Exiting")
    return None
