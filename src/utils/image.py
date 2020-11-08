from itertools import product
from typing import List

import numpy as np
from matplotlib import pyplot as plt


def show_generated_batch(images: List[np.array]) -> None:
    """Show batch of images with 2 rows and 4 columns"""
    figure, axes = plt.subplots(2, 4, figsize=[16, 8])

    for (x, y), image in zip(product(range(2), range(4)), images):
        axes[x, y].imshow(image[:, :, 0], cmap="Greys")
