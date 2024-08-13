import cv2
import numpy as np
from skimage.filters import threshold_sauvola


def sau(image,window,k):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    window_size =window
    thresh_sauvola = threshold_sauvola(image, window_size=window_size, k=k)
    binary_sauvola = image > thresh_sauvola
    uint8matrix = binary_sauvola.astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    uint8matrix = cv2.erode(uint8matrix, kernel, iterations=1)
    return uint8matrix
