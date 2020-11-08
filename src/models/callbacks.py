from itertools import product

import tensorflow as tf
from matplotlib import pyplot as plt


class GANCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(GANCallback, self).__init__()

    def on_epoch_end(self):
        ...

    @staticmethod
    def _show_generated_batch(images):
        figure, axes = plt.subplots(2, 4, figsize=[32, 12])

        for (x, y), image in zip(product(range(2), range(4)), images):
            axes[x, y].imshow(image[:, :, 0], cmap="Greys")
