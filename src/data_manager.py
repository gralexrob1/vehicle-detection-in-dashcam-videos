import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from skimage import io, transform


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

    fig, ax = plt.subplots(figsize=(10, 8))

    for x, y, dx, dy in bbs:

        rect = patches.Rectangle((x, y), dx, dy, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    ax.imshow(img)
    ax.set_title('Annotations for frame {}.'.format(frame))


def extract_annotations(path, df_annotation, frame):
    img = read_frame(path, df_annotation, frame)
    bbs = annotations_for_frame(df_annotation, frame)

    windows = []
    for x, y, dx, dy in bbs:
        window = img[y:y+dy, x:x+dx]
        window = transform.resize(window, (64,64))
        windows.append(window)
    
    return windows


def load_vehicle_data(data_path, folder_name):
    
    image_paths = []
    labels = []

    path = os.path.join(data_path, folder_name)

    for file in os.listdir(path):
        image_paths.append(os.path.join(path, file))
        labels.append(folder_name)

    return np.array(image_paths), np.array(labels)

