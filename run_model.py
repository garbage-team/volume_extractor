import numpy as np
from camera_controller import camera_capture
from point_cloud import xyz_to_volume, depth_to_xyz, bins_to_depth
from logger import log_depth_image
import os
if os.name == "nt":
    import tensorflow.lite as tflite
else:
    import tflite_runtime.interpreter as tflite


def run_model():
    """
    Runs an inference using a tflite model on an input image from a video capture stream

    :return: rgb image and depth image (float [1, h, w, c], float [1, h, w])
    """
    input_data = camera_capture(output_size=(224, 224),
                                normalize=True,
                                video_channel=0)
    rgb = np.expand_dims(input_data, 0)  # [h, w, c] -> [1(b), h, w, c]
    pred_depth = interpret_model(rgb,
                                 model_path="./lite_model.tflite")
    log_depth_image(pred_depth[0], rgb[0])
    return rgb, pred_depth


def interpret_model(input_data, model_path="./lite_model.tflite"):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    pred_depth = bins_to_depth(output_data)

    return pred_depth
