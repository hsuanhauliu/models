# Visualization Utils

import datetime

from PIL import Image
from typing import List, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np

def plot_imgs(imgs: List[Union[Image, np.ndarray]],
              titles: List[str]=[],
              fig_size: Tuple[int, int]=(10, 10)):
    """Plot multiple images easily.

    Keyword arguments:
    titles -- titles of each image to be plotted (default [])
    fig_size -- width and length of the final plotted image in inches (default (10, 10))
    """
    if (len(titles) != 0 and len(titles) != len(imgs)):
        raise ValueError("imgs and titles must be of same size")

    columns = 3 if len(imgs) > 3 else len(imgs)
    rows = (len(imgs) // columns) + (len(imgs) % columns > 0)
    plt.figure(figsize=fig_size)
    for i in range(len(imgs)):
        ax = plt.subplot(rows, columns, i + 1)
        plt.imshow(imgs[i])
        if (len(titles)):
            plt.title(titles[i])
        plt.axis("off")