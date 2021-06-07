from file_utils import read_rgb_image, read_depth_image
from point_cloud import depth_to_xyz, xyz_to_volume, clip_by_border
from model_functions import display_rgbd
from run_model import run_model
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


MODEL_PATH = "./lite_model_w_frl.tflite"

def main():
    cam_fov = (82.1, 52.2)
    vol_empty = -1.231
    fig, ax = plt.subplots(1,1)
    fig.suptitle('Garbage bin fill rate')
    plt.axis('off')
    img = ax.imshow(np.zeros((224,224)))
    img_text = ax.text(50,50,'Measuring..', color='red', fontsize=60)
    while True:
        rgb, depth = run_model(MODEL_PATH)
        depth = depth[0]
        rgb = rgb[0]
        xyz = depth_to_xyz(depth, fov=cam_fov)
        xyz_clipped = clip_by_border(xyz, (-1.4, 1.4), (-0.5, 0.75), (2.4, 1))
        xyz_clipped[:, 2] = 1.5 - xyz_clipped[:, 2]
        volume = xyz_to_volume(xyz_clipped)
        fill_rate = (1 - round(volume,3)/vol_empty)
        fill_rate_prc = round(fill_rate*100)
        print("Volume in container: ", volume, "m^3")
        display_image(fig, img, img_text, rgb, fill_rate_prc)
       	display_point_cloud(xyz_clipped)
    return None


def display_image(fig, im, text, rgb, fill_rate):
    if fill_rate < 75:
        txt_color = '#3ad20b' # Green if not full
    else:
        txt_color = 'red' # Red if needs to be emptied
    im.set_data(rgb)
    text.set_text(str(fill_rate)+"%")
    text.set_color(txt_color)
    fig.canvas.draw_idle()
    plt.pause(1)
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
