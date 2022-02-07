import math
import numpy as np

def get_vector(p1, p2):
    """
    Get the vector of two points
        inputs: two tuples of coordinates, p1, p2
        output: the tuple of the normalized vector, p1 -> p2, i.e., p2-p1
    """
    x = p2[0] - p1[0]
    y = p2[1] - p1[1]
    return (x, y)

def vector_normalization(p):
    """
    Get the normalized vector. Input is a vector, output is the normalized vector
    """
    l = math.sqrt(p[0]**2 + p[1]**2)
    return (p[0]/l, p[1]/l)


def get_bisector_vector(v1, v2):
    """
    Get the bisector of the two vectors
    """
    v1_normal = vector_normalization(v1)
    v2_normal = vector_normalization(v2)
    midpoint_x = (v1_normal[0] + v2_normal[0])/2
    midpoint_y = (v1_normal[1] + v2_normal[1])/2
    return (midpoint_x, midpoint_y)


def get_tightest_bounding_box(points):
    """
    Given point matrix where each row is a corner point of a polygon, get the tightest bounding box
    """
    xy_min = np.amin(points, axis=0)
    xy_max = np.amax(points, axis=0)

    return [xy_min[0], xy_max[0]], [xy_min[1], xy_max[1]]
