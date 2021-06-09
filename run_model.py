import numpy as np
import os
if os.name == "nt":  # If running Windows OS
    import tensorflow.lite as tflite
else:                # If anything else
    import tflite_runtime.interpreter as tflite


class DepthModel:
    def __init__(self, tflite_path):
        self.interpreter = tflite.Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']

    def predict(self, input_data):
        """
        Use the model to create a depth image prediction

        :param input_data: rgb data of shape [b, h, w, c]. If
        b > 1, the model will only return the first depth image
        :return: depth image of shape [b, h, w]
        """
        if self.input_shape != input_data.shape:
            raise TypeError("Wrong shape at input of model")
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        depth_bins = self.interpreter.get_tensor(self.output_details[0]['index'])
        depth_image = DepthModel._convert_to_depth(depth_bins)
        return depth_image

    @staticmethod
    def _convert_to_depth(depth_bins, anti_aliasing=True):
        """
        Function for converting the predicted softmax bins to
        an actual depth map.

        :param depth_bins: depth bins of shape [b, h, w, bins]. If
        b > 1, the function only returns the depth of the first entry
        in b.
        :param anti_aliasing: Applies anti-aliasing by interpolating
        between depths depending on the values of the bins
        :return: a depth map with shape [1, h, w]
        """
        if not anti_aliasing:
            depth_bins = np.argmax(depth_bins[0], axis=-1)  # -> [224, 224]
            depth_bins = one_hot(depth_bins, 150)
            depth_bins = np.expand_dims(depth_bins, axis=0)
        depth_image = DepthModel._bins_to_depth(depth_bins)
        return depth_image

    @staticmethod
    def _bins_to_depth(depth_bins):
        """
        Converts a bin array into a depth image
        Copy of src.image_utils bins_to_depth, but without tensorflow dependencies

        :param depth_bins: the depth bins in one_hot encoding, shape (b, h, w, c),
        the depth bins can also be passed as softmax bins of shape (b, h, w, c)
        :return: a depth image of shape (b, h, w)
        """
        bin_interval = (np.log10(80) - np.log10(0.25)) / 150
        # the borders variable here holds the depth for each specific value of the one hot encoded bins
        borders = np.asarray([np.log10(0.25) + (bin_interval * (i + 0.5)) for i in range(150)])
        depth = np.matmul(depth_bins, borders)  # [b, h, w, (c] * [c), 1] -> [b, h, w, 1]
        depth = np.power(10., depth)
        return depth

    def __call__(self, input_data):
        return self.predict(input_data)


def one_hot(index_array, max_value):
    """
    Helper function. creates a one-hot array based on the index_array

    :param index_array: indexes in the form of a numpy array
    :param max_value: depth of the one-hot output
    :return: one-hot array
    """
    one_hot_array = np.zeros((*index_array.shape, max_value))
    for i in np.arange(224):
        for j in np.arange(224):
            one_hot_array[i, j, index_array[i, j]] = 1
    return one_hot_array
