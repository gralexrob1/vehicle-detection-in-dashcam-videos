import numpy as np
from skimage import io, transform, feature
from tqdm import tqdm


def spatial_bin(image, size):

    features = transform.resize(image, size).ravel() 
    
    return features


def color_hist(image, nbins, bins_range):

    r_hist, _ = np.histogram(image[:,:,0], bins=nbins, range=bins_range)
    g_hist, _ = np.histogram(image[:,:,1], bins=nbins, range=bins_range)
    b_hist, _ = np.histogram(image[:,:,2], bins=nbins, range=bins_range)

    hist_features = np.concatenate((r_hist, g_hist, b_hist))

    return hist_features


def hog_hist(
        image, 
        orientations, pixels_per_cell, cells_per_block, 
        transform_sqrt, channel_axis=None
):
    features = feature.hog(
        image,
        orientations=orientations, 
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block, 
        transform_sqrt=transform_sqrt,
        channel_axis=channel_axis,
    )
    return features


def feature_extraction(
    image, 
    spatial_size,
    color_bins_n, color_bins_range,
    orientations, pixels_per_cell, cells_per_block, transform_sqrt,
    channel_axis=None
):
    
    features = []

    spatial_features = spatial_bin(
        image, 
        size=spatial_size
    )
    features.append(spatial_features)

    color_features = color_hist(
        image, 
        nbins=color_bins_n, bins_range=color_bins_range
    )
    features.append(color_features)

    if channel_axis is not None:
        for channel in range(image.shape[channel_axis]):
            hog_features = hog_hist(
                image[:,:,channel],
                orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block,
                transform_sqrt=transform_sqrt
            )
            features.append(hog_features)
    else:
        hog_features = hog_hist(
                image,
                orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block,
                transform_sqrt=transform_sqrt
            )
        features.append(hog_features)

    return np.concatenate(features)


def feature_pipeline(
    image_paths, 
    spatial_size,
    color_bins_n, color_bins_range,
    orientations, pixels_per_cell, cells_per_block, transform_sqrt,
    channel_axis=None
):

    features = []
    
    for image_path in tqdm(image_paths):
        frame = io.imread(image_path)
        frame_features = feature_extraction(
            frame,
            spatial_size=spatial_size,
            color_bins_n=color_bins_n, color_bins_range=color_bins_range,
            orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block,
            transform_sqrt=transform_sqrt, channel_axis=channel_axis
        )
        features.append(frame_features)
        
    features = np.array(features)

    return features


def feature_from_images_pipeline(
    images, 
    spatial_size,
    color_bins_n, color_bins_range,
    orientations, pixels_per_cell, cells_per_block, transform_sqrt,
    channel_axis=None
):

    features = []
    
    for image in tqdm(images):
        image_features = feature_extraction(
            image,
            spatial_size=spatial_size,
            color_bins_n=color_bins_n, color_bins_range=color_bins_range,
            orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block,
            transform_sqrt=transform_sqrt, channel_axis=channel_axis
        )
        features.append(image_features)

    return np.array(features)