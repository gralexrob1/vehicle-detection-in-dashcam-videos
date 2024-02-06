import cv2
import numpy as np


def slide_window_helper(img, x_start_stop=[None, None], y_start_stop=[None, None], window_size=[96, 64], overlapping=0.5):
    """
    Generate sliding windows over an image.

    Parameters:
    - img: numpy array, input image.
    - x_start_stop: list, x-axis start and stop positions for the window.
    - y_start_stop: list, y-axis start and stop positions for the window.
    - window_size: list, size of the sliding window [width, height].
    - overlapping: float, overlap percentage between consecutive windows.

    Returns:
    - windows: list, a list of sliding windows.
    """

    # If x or y start/stop positions are not defined, use the entire image
    x_start = x_start_stop[0] if x_start_stop[0] is not None else 0
    x_stop = x_start_stop[1] if x_start_stop[1] is not None else img.shape[1]
    y_start = y_start_stop[0] if y_start_stop[0] is not None else 0
    y_stop = y_start_stop[1] if y_start_stop[1] is not None else img.shape[0]

    # Calculate step size based on overlapping percentage
    step_x = int(window_size[0] * (1 - overlapping))
    step_y = int(window_size[1] * (1 - overlapping))

    # Initialize list to store sliding windows
    windows = []

    # Loop through x and y ranges
    for y in range(y_start, y_stop - window_size[1] + 1, step_y):
        for x in range(x_start, x_stop - window_size[0] + 1, step_x):
            # Calculate window coordinates
            window = [(x, y), (x + window_size[0], y + window_size[1])]
            windows.append(window)

    return windows


def slide_window(img, window_sizes, overlapping, x_start_stop=[None, None], y_start_stop=[None, None]):
    all_windows = []
    for (w, h), overlap in zip(window_sizes, overlapping):
        windows = slide_window_helper(img, x_start_stop, y_start_stop, [w, h], overlap)
        all_windows.extend(windows)
    return all_windows


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    imcopy = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    return imcopy