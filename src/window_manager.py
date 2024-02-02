import cv2
import numpy as np


# def slide_window_helper(
#     img, 
#     x_start_stop=[None, None], y_start_stop=[None, None], 
#     window_size=[96, 64]
# ):
    
#     window_size_x = window_size[0]
#     window_size_y = window_size[1]
#     xy_overlap=(0.5, 0.5)

#     # If x and/or y start/stop positions not defined, set to image size
#     if x_start_stop[0] == None:
#         x_start_stop[0] = 0
#     if x_start_stop[1] == None:
#         x_start_stop[1] = img.shape[1]
#     if y_start_stop[0] == None:
#         y_start_stop[0] = 0
#     if y_start_stop[1] == None:
#         y_start_stop[1] = img.shape[0]

#     # Compute the span of the region to be searched    
#     xspan = x_start_stop[1] - x_start_stop[0]
#     yspan = y_start_stop[1] - y_start_stop[0]

#     # Compute the number of pixels per step in x/y
#     nx_pix_per_step = np.int32(window_size_x*(1 - xy_overlap[0]))
#     ny_pix_per_step = np.int32(window_size_y*(1 - xy_overlap[1]))

#     # Compute the number of windows in x/y
#     nx_windows = np.int32(xspan/nx_pix_per_step) - 2
#     ny_windows = np.int32(yspan/ny_pix_per_step) - 2

#     # Initialize a list to append window positions to
#     window_list = []
    
#     ys = y_start_stop[0]
#     while ys + window_size_y < y_start_stop[1]: 

#         xs = x_start_stop[0]
#         while xs < x_start_stop[1]:
#             # Calculate window position
#             endx = xs + window_size_x
#             endy = ys + window_size_y

#             # Append window position to list
#             window_list.append(((xs, ys), (endx, endy)))

#             xs += nx_pix_per_step

#         window_size_x = int(window_size_x * 1.3)
#         window_size_y = int(window_size_y * 1.3)
#         nx_pix_per_step = np.int32(window_size_x*(1 - xy_overlap[0]))
#         ny_pix_per_step = np.int32(window_size_y*(1 - xy_overlap[1]))
#         ys += ny_pix_per_step

#     return window_list


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


# def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None]):
#     windows_a = slide_window_helper(img, x_start_stop, y_start_stop, window_size=[128, 128])
#     windows_b = slide_window_helper(img, x_start_stop, y_start_stop, window_size=[64, 64])
#     windows_c = slide_window_helper(img, x_start_stop, y_start_stop, window_size=[32, 32])
#     return windows_a + windows_b + windows_c

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