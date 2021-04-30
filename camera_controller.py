import cv2
import numpy as np
from math import pi


def camera_capture(output_size=(224, 224), normalize=True, video_channel=0, crop_fov=(69.4, 42.5)):
    cap = cv2.VideoCapture(video_channel)
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = crop_to_fov(frame, crop_fov)
    frame = cv2.resize(frame, output_size)
    if normalize:
        frame = frame.astype('float32')
        frame = frame / 255.0
    return frame


def crop_to_fov(frame, fov):
    x_fov, x_in_fov = fov[0], 82.1
    y_fov, y_in_fov = fov[1], 52.2
    x_factor = np.sin((x_fov * pi) / 180) / np.sin((x_in_fov * pi) / 180)
    y_factor = np.sin((y_fov * pi) / 180) / np.sin((y_in_fov * pi) / 180)
    new_size = (int(frame.shape[0] * x_factor), int(frame.shape[1] * y_factor))
    offset_x = (frame.shape[0] - new_size[0]) // 2
    offset_y = (frame.shape[1] - new_size[1]) // 2
    out_frame = frame[offset_x:(offset_x + new_size[0]),
                      offset_y:(offset_y + new_size[1])]
    return out_frame
