from PointCloud import PointCloud
from run_model import DepthModel
from camera_controller import CameraInterface
from config import read_config, save_config
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


MODEL_PATH = "./lite_model_w_frl.tflite"


def main():
    cfg = read_config()
    depth_model = DepthModel(cfg["model_path"])
    camera = CameraInterface((cfg["var_webcam_fov_x"], cfg["var_webcam_fov_y"]),
                             (cfg["var_crop_fov_x"], cfg["var_crop_fov_y"]),
                             (224, 224))
    fig, ax = plt.subplots(1, 1)
    fig.suptitle('Garbage bin fill rate')
    plt.axis('off')
    img = ax.imshow(np.zeros((224, 224)))
    img_text = ax.text(50, 50, 'Measuring..',
                       color='red',
                       fontsize=60)
    while True:
        rgb = camera.capture_image()
        rgb = np.expand_dims(rgb, axis=0)  # Expand the first dimension to fit tflite model
        depth = depth_model(rgb)
        depth = depth[0]    # Remove batch dimension
        rgb = rgb[0]        # Remove batch dimension
        pc = PointCloud.from_depth(depth, fov=camera.output_fov)
        pc = pc.select_roi(borders=np.asarray([[-1.4, 1.40],
                                               [-0.5, 0.75],
                                               [2.4,  1.00]]))
        pc[:, 2] = 1.5 - pc[:, 2]
        volume = pc.to_volume()
        fill_rate = (1 - round(volume, 3)/cfg["var_empty_volume"])
        fill_rate_prc = round(fill_rate*100)
        print("Volume in container: ", volume, "m^3")
        display_image(fig, img, img_text, rgb, fill_rate_prc)
        display_point_cloud(pc)


def display_image(fig, im, text, rgb, fill_rate):
    if fill_rate < 75:
        txt_color = '#3ad20b'   # Green if not full
    else:
        txt_color = 'red'       # Red if needs to be emptied
    im.set_data(rgb)
    text.set_text(str(fill_rate)+"%")
    text.set_color(txt_color)
    fig.canvas.draw_idle()
    plt.pause(1)
    return None


def display_point_cloud(xyz):
    _ = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter3D(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=xyz[:, 2], cmap='hsv')
    ax.set_title("3D plot")
    plt.show()
    return None


if __name__ == '__main__':
    main()
