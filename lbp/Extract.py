
import numpy as np
from Color import rgb2hsv, rgb2ycbcr, rgb2yuv
from lbp import lbp_basic, lbp_uniform, lbp_rotation_invariant, lbp_multi_scale


def convert_to_color_space(image, color_space):
    if color_space == 'HSV':
        return rgb2hsv(image)
    elif color_space == 'YCbCr':
        return rgb2ycbcr(image)
    elif color_space == 'YUV':
        return rgb2yuv(image)
    elif color_space == 'RGB':
        return image
    else:
        raise ValueError(f"Unsupported color space: {color_space}")


def extract_lbp_features(image, method='basic', color_space='RGB', P=8, R=1, P_list=[8], R_list=[1]):
    color_image = convert_to_color_space(image, color_space)

    features = []
    for i in range(color_image.shape[2]):
        channel = color_image[..., i]

        if method == 'basic':
            hist = lbp_basic(channel, P, R)
        elif method == 'uniform':
            hist = lbp_uniform(channel, P, R)
        elif method == 'rotation_invariant':
            hist = lbp_rotation_invariant(channel, P, R)
        elif method == 'multi_scale':
            hist = lbp_multi_scale(channel, P_list, R_list)
        else:
            raise ValueError(f"Unsupported LBP method: {method}")

        features.append(hist)
    feature_vector = np.concatenate(features)

    return feature_vector
