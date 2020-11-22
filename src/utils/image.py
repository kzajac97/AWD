from itertools import product
from pathlib import Path
from typing import List, Union

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def show_batch(images: List[np.array]) -> None:
    """Show batch of images with 2 rows and 4 columns"""
    figure, axes = plt.subplots(2, 4, figsize=[16, 8])

    for (x, y), image in zip(product(range(2), range(4)), images):
        axes[x, y].imshow(image[:, :, 0], cmap="Greys")


def read_image_as_tensor(path: Union[str, Path], target_dim: tuple) -> tf.Tensor:
    """Load image as VGG19 compatible tensor"""
    image = tf.keras.preprocessing.image.load_img(path, target_size=target_dim)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.vgg19.preprocess_input(image)

    return image


def tensor_to_image(tensor: tf.Tensor, width: int, height: int, channels: int = 3) -> np.array:
    """
    Convert tensor representation of an image into displayable array

    :param tensor: tensor with image content
    :param height: target image height
    :param width: target image width
    :param channels: number of channels in an image

    :return: ND numpy array with image content
    """
    tensor = tensor.reshape((height, width, channels))
    # Remove zero-center by mean pixel
    tensor[:, :, 0] += 103.939
    tensor[:, :, 1] += 116.779
    tensor[:, :, 2] += 123.68

    tensor = tensor[:, :, ::-1]
    return np.clip(tensor, 0, 255).astype('uint8')
