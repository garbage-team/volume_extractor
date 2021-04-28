from file_utils import write_depth_image, write_rgb_image


def log_depth_image(depth, rgb):
    """
    Logs depth image into a file

    Writes binary values of the depth matrix into a raw file and saves
    the shape information of the matrix into a .csv file with the format
    "[height (int)],[width (int)][newline]"

    Also saves the rgb image in a png file

    There needs to be a folder named log in the folder where the script
    is run from

    :param depth: a numpy depth matrix of int16 values of shape [h, w]
    :return: True
    """
    # Save raw data into a depth .raw file
    path = "./log/last_depth.raw"
    values = depth * 1000
    write_depth_image(values.astype('uint16'), path)

    # Save the rgb image in a png file
    rgb = (rgb*255).astype('uint8')
    path = "./log/last_rgb.png"
    write_rgb_image(rgb, path)
    return True
