from data_manager import load_vehicle_data
from feature_manager import feature_pipeline

import numpy as np
from sklearn.preprocessing import StandardScaler


def preprocessing(
    data_path, folders,
    spatial_size,
    color_bins_n, color_bins_range,
    orientations, pixels_per_cell, cells_per_block, transform_sqrt,
    channel_axis=None,
    verbose=False
):
    
    image_paths = []
    labels = []

    for folder in folders:
        folder_image_paths, folder_labels = load_vehicle_data(data_path, folder)
        image_paths.extend(folder_image_paths)
        labels.extend(folder_labels)
    
    image_paths = np.array(image_paths)
    labels = np.array(labels)

    if verbose:
        print(f"Image paths shape: {image_paths.shape}")
        print(f"Labels shapes: {labels.shape}")

    X = feature_pipeline(
        image_paths, 
        spatial_size,
        color_bins_n, color_bins_range,
        orientations, pixels_per_cell, cells_per_block, transform_sqrt,
        channel_axis=channel_axis
    )
                   
    X_scaler = StandardScaler().fit(X)
    X_scaled = X_scaler.transform(X)

    y = np.array([1 if label=='vehicles' else 0 for label in labels])
    
    return X_scaled, X_scaler, y 