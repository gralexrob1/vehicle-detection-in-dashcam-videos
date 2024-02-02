import cv2
import matplotlib.pyplot as plt
import numpy as np


def compute_gradients(image):
    """
    Computes gradient magnitude and gradient direction of an image
    based on Sobel filters.

    Parameters
    ----------
    image:
        Input image of size (h, w)

    Returns
    -------
    magnitude:
        Image of size (h, w) with gradient magnitude
    direction:
        Image of size (h, w) with gradient direction
    """

    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1)

    magnitude, direction = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    return magnitude, direction



def compute_histograms(magnitude, direction, cell_size=(8,8), bins=9, plot=False):

    direction = np.mod(direction, 180)

    cells_x = magnitude.shape[1] // cell_size[0]
    cells_y = magnitude.shape[0] // cell_size[1]

    # print(cells_x) # 6
    # print(cells_y) # 6

    histograms = []

    for i in range(cells_y):
        for j in range(cells_x):

            cell_magnitude = magnitude[
                i * cell_size[1]: (i + 1) * cell_size[1], 
                j * cell_size[0]: (j + 1) * cell_size[0]
            ]
            cell_direction = direction[
                i * cell_size[1]: (i + 1) * cell_size[1], 
                j * cell_size[0]: (j + 1) * cell_size[0]
            ]

            histogram, _ = np.histogram(cell_direction, bins=bins, range=(0, 180), weights=cell_magnitude)
            histograms.append(histogram)

    histograms = np.array(histograms)

    if plot:

        fig, axs = plt.subplots(cells_x,  cells_y, figsize=(15, 15), sharex=True, sharey=True)

        for i in range(cells_y):
            for j in range(cells_x):
                axs[i, j].bar(np.arange(bins)*20, histograms[i*cells_y+j], width=20, align='edge')
                axs[i, j].set_title(f'Cell ({j}, {i})')
                axs[i, j].tick_params(axis='x', rotation=45)
        
        plt.xticks(np.arange(0, 10 * 20, 20))
        plt.tight_layout()         

    return histograms, cells_x, cells_y


def normalize_histograms(histograms, cells_x, cells_y, block_size=(2,2), plot=False):

    blocks_x = cells_y - block_size[0] + 1
    blocks_y = cells_x - block_size[1] + 1

    original_shape = histograms.shape
    histograms = histograms.reshape(cells_x, cells_y, histograms.shape[1])

    normalized_histograms = np.zeros_like(histograms)

    for i in range(blocks_y):
        for j in range(blocks_x):

            block = histograms[i:i+block_size[1], j:j+block_size[0], :]
            block_norm = np.linalg.norm(block)

            if block_norm != 0:
                normalized_block = block / block_norm
            else:
                normalized_block = block

            normalized_histograms[i:i+block_size[1], j:j+block_size[0], :] = normalized_block
    
    normalized_histograms = normalized_histograms.reshape(original_shape)

    if plot:

        histograms = normalized_histograms

        fig, axs = plt.subplots(cells_x,  cells_y, figsize=(15, 15), sharex=True, sharey=True)

        for i in range(cells_y):
            for j in range(cells_x):
                axs[i, j].bar(np.arange(9)*20, histograms[i*cells_y+j], width=20, align='edge')
                axs[i, j].set_title(f'Cell ({j}, {i})')
                axs[i, j].tick_params(axis='x', rotation=45)
        
        plt.xticks(np.arange(0, 10 * 20, 20))
        plt.tight_layout()         

    return normalized_histograms
