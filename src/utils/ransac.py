######################
# Code from following article and by article author:
# https://medium.com/mlearning-ai/recursive-ransac-approach-to-find-all-straight-lines-in-an-image-b5c510a0224a
# Some modification carried out
########################
import numpy as np
from skimage.color import rgb2gray
from skimage.measure import LineModelND, ransac
import cv2

def read_image_array_as_cartesian_data_points(np_image):
    np_image = rgb2gray(np_image)
    if (np_image.dtype == 'float'):
        black_white_threshold=0.5
    elif (np_image.dtype == 'uint8'):
        black_white_threshold=128
    else:
        raise Exception("Invalid dtype %s " % (np_image.dtype))
    indices=np.where(np_image <= black_white_threshold)
    width=np_image.shape[1]
    height=np_image.shape[0]
    cartesian_y=height-indices[0]-1
    np_data_points=np.column_stack((indices[1],cartesian_y))
    return np_data_points, width,height

def find_line(data_points:np.ndarray,width):
    """
    Find and draw a single ransac line
    :param data_points:
    :param width:
    :return: x and y coordinates of line
    """
    distance_from_line=2
    model_robust, inliers = ransac(data_points, LineModelND, min_samples=3,
                                   residual_threshold=distance_from_line, max_trials=1000)

    line_x = np.arange(0, width)
    line_y = model_robust.predict_y(line_x)
    return line_x, line_y

def extract_best_ransac_line(np_image,original, lower=False):
    """
    Extracts the best possible line using scikit learn's RANSAC function
    and creates a new image in with this line imposed on image.
    """
    image,width,height=read_image_array_as_cartesian_data_points(np_image)
    x, y=find_line(image,width)

    # For poly fill find the correct order to plot polygon
    if x[np.argmax(y)] > x[np.argmin(y)]:
        outer_y = int(original.shape[0] - max(y))
        inner_y = int(original.shape[0] - min(y))
    else:
        outer_y = int(original.shape[0] - min(y))
        inner_y = int(original.shape[0] - max(y))

    # Offset the line slightly so that field markings remain visible
    if lower == False:
        offset_factor = 0
    else:
        offset_factor = -0
    contours = np.array([[0, 0],
                         [original.shape[1], 0],
                         [original.shape[1], outer_y-offset_factor],
                         [0, inner_y-offset_factor]])

    contours_lower = np.array([[0, inner_y - offset_factor],
                         [original.shape[1], outer_y - offset_factor],
                         [original.shape[1], original.shape[0]],
                         [0, original.shape[0]]])
    # Polyfill above or below line depending on which sideline is found
    if lower == False:
        cv2.fillPoly(original, pts=[contours], color=(0, 0, 0))
        contours_used = contours
    else:
        cv2.fillPoly(original, pts=[contours_lower], color=(0, 0, 0))
        contours_used = contours_lower
    return original, contours_used

