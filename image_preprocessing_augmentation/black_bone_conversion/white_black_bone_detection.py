import numpy as np
import cv2
import skimage
import skimage.filters


def horizontal_edge_detector(image, kernel_size=5, low_threshold=0.5, high_threshold=0.8):
    """
    Given an image, detect the horizontal edges
    """
    edges = cv2.Sobel(image, cv2.CV_16U, 0, 1, ksize=kernel_size)
    # sobely = np.multiply((sobely >  threshold), 1)
    # edges = skimage.filters.sobel_h(image)
    edges = np.array(edges, dtype=float) / (2 ** 16 -1)
    hyst = skimage.filters.apply_hysteresis_threshold(edges, low=low_threshold, high=high_threshold)
    return hyst


def black_white_bone_checker(image, num_pixels_near_edge_to_image_size=0.05, kernel_size=5, low_threshold=0.6, high_threshold=0.8):
    """
    Check whether the given image is inverted
    """
    image = np.array(image, dtype=float)
    if len(image.shape) == 3:
        index = image.shape.index(1)
        if index == 0:
            image = image[0, :, :]
        elif index == 1:
            image = image[:, 0, :]
        else:
            image = image[:, :, 0]
    num_pixels_near_edge = int(image.shape[0] * num_pixels_near_edge_to_image_size)
    edges_detected_image = horizontal_edge_detector(image, kernel_size=kernel_size, low_threshold=low_threshold, high_threshold=high_threshold)
    img_height, img_width = image.shape
    edges_detected_image = edges_detected_image[num_pixels_near_edge:img_height-num_pixels_near_edge, :]
    edge_points_row_indices, edge_points_col_indices = np.where(edges_detected_image)
    edge_points_row_indices = np.array(edge_points_row_indices)
    edge_points_col_indices = np.array(edge_points_col_indices)
    steps = np.ones_like(edge_points_row_indices)
    near_edge_rows = steps[:, None] * np.arange(2 * num_pixels_near_edge) + edge_points_row_indices[:, None]
    near_edge_cols = np.ones_like(near_edge_rows) * edge_points_col_indices[:, None]

    near_edge_pixels = image[near_edge_rows, near_edge_cols]
    edges = image[edge_points_row_indices + num_pixels_near_edge, edge_points_col_indices] 
    min_near_edge_pixels = np.min(near_edge_pixels, axis=1)
    max_near_edge_pixels = np.max(near_edge_pixels, axis=1)
    to_min_distances = edges - min_near_edge_pixels
    to_max_distances = max_near_edge_pixels - edges
    mean_min_distance = np.mean(to_min_distances)
    mean_max_distance = np.mean(to_max_distances)

    if mean_min_distance < mean_max_distance:
        rst = 'black'
    else:
        rst = 'white'

    return rst, mean_min_distance, mean_max_distance
