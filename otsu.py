from cv2 import THRESH_BINARY
import numpy as np
import cv2


def otsu(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bins_num = 256
    hist, bin_edges = np.histogram(image, bins=bins_num)
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    mean1 = np.cumsum(hist * bin_mids) / weight1
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]
    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    index_of_max_val = np.argmax(inter_class_variance)
    threshold = bin_mids[:-1][index_of_max_val]
    otsu_threshold, image_result = cv2.threshold(image, threshold, 255, THRESH_BINARY)
    return otsu_threshold
