
import numpy as np

def rgb2hsv(image):
    image = image / 255.0
    R, G, B = image[..., 0], image[..., 1], image[..., 2]
    Cmax = np.max(image, axis=-1)
    Cmin = np.min(image, axis=-1)
    delta = Cmax - Cmin
    hue = np.zeros_like(Cmax)
    hue[Cmax == R] = (G[Cmax == R] - B[Cmax == R]) / delta[Cmax == R]
    hue[Cmax == G] = (B[Cmax == G] - R[Cmax == G]) / delta[Cmax == G] + 2
    hue[Cmax == B] = (R[Cmax == B] - G[Cmax == B]) / delta[Cmax == B] + 4
    hue = (hue / 6.0) % 1.0
    saturation = np.zeros_like(Cmax)
    saturation[Cmax != 0] = delta[Cmax != 0] / Cmax[Cmax != 0]
    value = Cmax

    hsv = np.stack([hue, saturation, value], axis=-1)
    return hsv

def rgb2ycbcr(image):
    image = image / 255.0
    R, G, B = image[..., 0], image[..., 1], image[..., 2]

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 0.5
    Cr = 0.5 * R - 0.460593 * G - 0.039407 * B + 0.5

    ycbcr = np.stack([Y, Cb, Cr], axis=-1)
    return ycbcr

def rgb2yuv(image):
    image = image / 255.0
    R, G, B = image[..., 0], image[..., 1], image[..., 2]

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    U = -0.14713 * R - 0.28886 * G + 0.436 * B
    V = 0.615 * R - 0.51499 * G - 0.10001 * B

    yuv = np.stack([Y, U, V], axis=-1)
    return yuv
