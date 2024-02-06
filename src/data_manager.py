import os
import math
import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from tqdm import tqdm
from skimage import io, color, transform


# FROM BASELINE NOTEBOOK

def read_frame(path, df_annotation, frame):
    """Read frames and create integer frame_id-s"""
    file_path = df_annotation[df_annotation.index == frame]['frame_id'].values[0]
    return io.imread(os.path.join(path, file_path))


def annotations_for_frame(df_annotation, frame):
    assert frame in df_annotation.index
    bbs = df_annotation[df_annotation.index == frame].bounding_boxes.values[0]
    bbs = str(bbs).split(' ')
    if len(bbs)<4:
        return []

    bbs = list(map(lambda x : int(x),bbs))

    return np.array_split(bbs, len(bbs) / 4)


def show_annotation(path, df_annotation, frame):
    img = read_frame(path, df_annotation, frame)
    bbs = annotations_for_frame(df_annotation, frame)

    fig, ax = plt.subplots(figsize=(16, 9))

    for x, y, dx, dy in bbs:

        rect = patches.Rectangle((x, y), dx, dy, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    ax.imshow(img)
    ax.set_title('Annotations for frame {}.'.format(frame))


def bounding_boxes_to_mask(bounding_boxes, H, W):
    
    """
    Converts set of bounding boxes to a binary mask
    """

    mask = np.zeros((H, W))
    for x, y, dx, dy in bounding_boxes:
        mask[y:y+dy, x:x+dx] = 1

    return mask


# BUILD VEHICLE DATASET

def extract_annotations(path, df_annotation, frame, resize=(64,64), gray=False):
    img = read_frame(path, df_annotation, frame)
    bbs = annotations_for_frame(df_annotation, frame)

    windows = []
    for x, y, dx, dy in bbs:
        window = img[y:y+dy, x:x+dx]
        window = transform.resize(window, resize)
        if gray:
            window = color.rgb2gray(window)
        windows.append(window)
    
    return windows


# BUILD NON-VEHICLE DATASET

def round_to_power_of_two(x):
    return 2 ** round(math.log2(x))


def add_rectangle_to_mask(mask, rect_coords, value=1):
    x, y, w, h = rect_coords
    mask[y:y+h, x:x+w] = value
    return mask


def build_non_vehicle_rectangles(
    vehicle_rectangles,
    frame_size=(1280, 720),
    vertical_bounds=[None, None],
    max_attempts=1000,
    rectangle_default_size=(64,64)
):

    non_vehicle_rectangles = []

    vehicle_mask = bounding_boxes_to_mask(vehicle_rectangles, H=frame_size[1], W=frame_size[0])
    mask_to_avoid = vehicle_mask.copy()

    for _, _, dx, dy in vehicle_rectangles:

        dx = round_to_power_of_two(dx)
        dy = round_to_power_of_two(dy)

        loop_count=0
        while True:
            loop_count+=1

            x_non_vehicle = random.randint(0, frame_size[0] - dx)

            if vertical_bounds[0] is not None and vertical_bounds[1] is not None:
                y_non_vehicle = random.randint(vertical_bounds[0], vertical_bounds[1] - dy)
            elif vertical_bounds[0] is not None:
                y_non_vehicle = random.randint(vertical_bounds[0], frame_size[1] - dy)
            elif vertical_bounds[1] is not None:
                y_non_vehicle = random.randint(0, vertical_bounds[1] - dy)
            else:
                y_non_vehicle = random.randint(0, frame_size[1] - dy)

            if np.sum(mask_to_avoid[
                y_non_vehicle:y_non_vehicle + dy, 
                x_non_vehicle:x_non_vehicle + dx
            ]) == 0:
                non_vehicle_rectangles.append(np.array([x_non_vehicle, y_non_vehicle, dx, dy]))
                mask_to_avoid = add_rectangle_to_mask(mask_to_avoid, np.array([x_non_vehicle, y_non_vehicle, dx, dy]))
                break

            if loop_count==max_attempts:
                    dx, dy = rectangle_default_size
    
    return non_vehicle_rectangles


def extract_rectangles(image, rectangles, resize=(64,64)):

    windows = []
    for x, y, dx, dy in rectangles:
        window = image[y:y+dy, x:x+dx]
        window = transform.resize(window, resize)
        windows.append(window)
    
    return windows


# BUILD CLASSIFICATION DATASET

def build_classification_dataset(
    path, df_annotation, 
    frame_size=(1280,720), vertical_bounds=[None, None],
    max_attempts=1000, rectangle_default_size=(64,64)
):
    
    vehicle_images = []
    non_vehicle_images = []

    for line in tqdm(range(len(df_annotation))):

        frame = read_frame(path, df_annotation, line)

        # VEHICLE DATASET

        vehicle_rectangles = annotations_for_frame(df_annotation, line)
        vehicle_windows = extract_rectangles(frame, vehicle_rectangles, resize=rectangle_default_size)
        vehicle_images.extend(vehicle_windows)

        # NON-VEHICLE DATASET

        non_vehicle_rectangles = build_non_vehicle_rectangles(
            vehicle_rectangles,
            frame_size=frame_size,
            vertical_bounds=vertical_bounds,
            max_attempts=max_attempts,
            rectangle_default_size=rectangle_default_size
        ) 
        non_vehicle_windows = extract_rectangles(frame, non_vehicle_rectangles, resize=rectangle_default_size)
        non_vehicle_images.extend(non_vehicle_windows)

    vehicle_labels = np.ones(len(vehicle_images))
    non_vehicle_labels = np.zeros(len(non_vehicle_images))

    images = np.concatenate([vehicle_images, non_vehicle_images])
    labels = np.concatenate([vehicle_labels, non_vehicle_labels])

    return np.array(images), np.array(labels)


# OTHER

def run_length_encoding(mask):

    """
    Produces run length encoding for a given binary mask
    """
    
    # find mask non-zeros in flattened representation
    non_zeros = np.nonzero(mask.flatten())[0]
    padded = np.pad(non_zeros, pad_width=1, mode='edge')
    
    # find start and end points of non-zeros runs
    limits = (padded[1:] - padded[:-1]) != 1
    starts = non_zeros[limits[:-1]]
    ends = non_zeros[limits[1:]]
    lengths = ends - starts + 1

    return ' '.join(['%d %d' % (s, l) for s, l in zip(starts, lengths)])


def load_vehicle_data(data_path, folder_name):
    
    image_paths = []
    labels = []

    path = os.path.join(data_path, folder_name)

    for file in os.listdir(path):
        image_paths.append(os.path.join(path, file))
        labels.append(folder_name)

    return np.array(image_paths), np.array(labels)

