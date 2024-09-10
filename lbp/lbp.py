import numpy as np
def compute_lbp(image, P=8, R=1):

    rows, cols = image.shape
    lbp_image = np.zeros_like(image, dtype=np.uint8)

    for i in range(R, rows - R):
        for j in range(R, cols - R):
            center = image[i, j]
            binary_string = ""
            for p in range(P):
                theta = 2 * np.pi * p / P
                x = i + R * np.sin(theta)
                y = j + R * np.cos(theta)
                x, y = int(round(x)), int(round(y))
                binary_string += str(image[x, y] >= center)
            lbp_image[i, j] = int(binary_string, 2)

    return lbp_image


def lbp_basic(image, P=8, R=1):
    lbp_image = compute_lbp(image, P, R)
    n_bins = 2 ** P
    hist, _ = np.histogram(lbp_image, bins=n_bins, range=(0, n_bins))
    return hist


def lbp_uniform(image, P=8, R=1):
    lbp_image = compute_lbp(image, P, R)
    n_bins = P + 2

    uniform_patterns = np.zeros(n_bins)
    for i in range(n_bins):
        if bin(i).count('1') <= 2:
            uniform_patterns[i] = 1

    lbp_hist, _ = np.histogram(lbp_image, bins=n_bins, range=(0, n_bins))
    return lbp_hist


def lbp_rotation_invariant(image, P=8, R=1):
    lbp_image = compute_lbp(image, P, R)
    n_bins = 2 ** P
    lbp_hist = np.zeros(n_bins)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            lbp_value = lbp_image[i, j]
            min_rotation_value = lbp_value
            for k in range(P):
                rotated_value = ((lbp_value << k) | (lbp_value >> (P - k))) & (n_bins - 1)
                min_rotation_value = min(min_rotation_value, rotated_value)
            lbp_hist[min_rotation_value] += 1

    return lbp_hist

def lbp_multi_scale(image, P_list=[8], R_list=[1]):
    features = []
    for P, R in zip(P_list, R_list):
        lbp_image = compute_lbp(image, P, R)
        hist, _ = np.histogram(lbp_image, bins=2 ** P, range=(0, 2 ** P))
        features.append(hist)
    return np.concatenate(features)
