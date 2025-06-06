import tifffile
import skimage
import numpy as np
import torch


def rgb2gray(x):
    """
    Reads numpy image in the original shape and converts it to grayscale.
    """
    return np.mean(x, axis=-1)


def read_file(input_path):
    path = input_path.name

    if ('.tif' in path) or ('.tiff' in path):
        im = tifffile.imread(path)
    else:
        im = skimage.io.imread(path)

    im = np.array(im)

    if len(im.shape) == 2:
        # "Grayscale Image"
        image_type = 0
    elif len(im.shape) == 3 and im.shape[2] == 3:
        # "RGB Image"
        image_type = 1
        im = rgb2gray(im)
    elif len(im.shape) == 3:
        # "Volumetric Data or Multi-Channel Image"
        image_type = 2
    else:
        # "Unknown Data Type"
        image_type = -1

    return im, image_type