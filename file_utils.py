"""
Module for saving and loading files relevant for the project

depth images, and such
"""
import struct
import cv2
import numpy as np


def read_depth_image(path, shape=(480, 640)):
    with open(path, 'rb') as file:
        depth_image = np.asarray(struct.unpack('H' * np.prod(shape), file.read())).reshape(shape)
    return depth_image


def write_depth_image(depth, path):
    """
    Writes a uint16 array depth to a binary file at path

    :param depth: uint16 array
    :param path: path to the file where the depth image is stored (recommended format: [filename].raw)
    :return: True if everything went well
    """
    with open(path, 'wb') as file:
        file.write(
            struct.pack(
                'H' * np.prod(depth.shape),
                *depth.flatten().tolist()
            )
        )
    return True


def read_rgb_image(path):
    rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return rgb


def write_rgb_image(rgb, path):
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return True
