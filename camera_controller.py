import cv2
import numpy as np
from math import pi


class CameraInterface:
    def __init__(self, input_fov, output_fov, output_size, video_channel=0):
        self.capture_stream = cv2.VideoCapture(video_channel)
        self.input_fov = input_fov
        self.output_fov = output_fov
        self.output_size = output_size

    def capture_image(self, crop_to_fov=True, normalize=True, empty_buffer=True):
        if empty_buffer:
            for i in range(10):  # Empty capture buffer, get fresh image
                _, _ = self.capture_stream.read()
                cv2.waitKey(1)
        ret, frame = self.capture_stream.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if crop_to_fov:     # Narrow the field of view of the image
            frame = self._crop_to_fov(frame)
        if normalize:       # Divide all the pixel values with 255 to normalize
            frame = frame.astype("float32")  # Convert from uint8 to float32
            frame = frame / 255.
        # Finally resize the image to given size
        frame = cv2.resize(frame, self.output_size)
        return frame

    def _crop_to_fov(self, frame):
        x_fov, x_in_fov = self.output_fov[0], self.input_fov[0]
        y_fov, y_in_fov = self.output_fov[1], self.input_fov[1]
        x_factor = np.sin((x_fov * pi) / 180) / np.sin((x_in_fov * pi) / 180)
        y_factor = np.sin((y_fov * pi) / 180) / np.sin((y_in_fov * pi) / 180)
        new_size = (int(frame.shape[0] * x_factor), int(frame.shape[1] * y_factor))
        offset_x = (frame.shape[0] - new_size[0]) // 2
        offset_y = (frame.shape[1] - new_size[1]) // 2
        out_frame = frame[offset_x:(offset_x + new_size[0]),
                          offset_y:(offset_y + new_size[1])]
        return out_frame
