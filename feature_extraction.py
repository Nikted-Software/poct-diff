import cv2
from scipy.signal import find_peaks
from otsu import otsu
import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
from threshold_sauvola import sau
import math

warnings.filterwarnings("ignore", message="invalid value encountered in divide")

def estimation(image1,calibration_coefficient,thickness,s_area):

    # local threshold with small window to find order of wbc number
    window_size = 15
    sau_threshold = -0.05
    minimum_size = 1

    image11 = sau(image1, window_size, sau_threshold)
    contours, hierarchy = cv2.findContours(
        image11, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )
    count = 0
    for c in contours:
        if cv2.contourArea(c) >= 16 and cv2.contourArea(c) <= 500:
            count += 1
    estimation = int(calibration_coefficient * count / (thickness * s_area))
    print("estimation:", estimation)
    if estimation > 8000:
        maximum_size = 500
    else:
        maximum_size = 300
    return maximum_size

def noise_recognition(image1,maximum_size):

    # local threshold with big window to recognize stains and noises
    image2 = image1
    # image3 = image1.copy()
    maximum_length = 100
    thresh = sau(image1, 1001, -0.6)
    cv2.imwrite("big_window.jpg", thresh)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont = []
    for contour in contours:
        cnt = contour
        leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
        rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
        topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
        bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
        if (
            cv2.contourArea(contour) >= maximum_size
            or abs(rightmost[0] - leftmost[0]) > maximum_length
            or abs(topmost[1] - bottommost[1]) > maximum_length
        ):
            cont.append(contour)
            cv2.drawContours(image2, [contour], 0, (0, 0, 255), 2)
    cv2.imwrite("big_contour.jpg", image2)

    # Determine whether there is a channel in the image or not
    largest_contour = max(contours, key=cv2.contourArea)
    cnt = largest_contour
    leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
    rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
    topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
    bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
    if (
        cv2.arcLength(largest_contour, True) > 2000
        or abs(rightmost[0] - leftmost[0]) > 500
        or abs(topmost[1] - bottommost[1]) > 500
    ):
        print("channel")
    return cont
def green_and_size_threshold_finder(image1,cont,maximum_size):
    # local threshold with small window to find size and green threshold
    window_size = 15
    sau_threshold = -0.05
    minimum_size = 1

    image11 = sau(image1, window_size, sau_threshold)
    contours, hierarchy = cv2.findContours(
        image11, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )
    cv2.imwrite("small_window.jpg", image11)
    count = 0
    p = 0
    thf = []
    are = []
    le = len(cont)
    for c in contours:
        xx, yy, ww, hh = cv2.boundingRect(c)
        cr = image1[int(yy - p) : int(yy + hh + p), int(xx - p) : int(xx + ww + p)]
        count += 1
        thf0 = []
        if le != 0:
            if (
                cv2.contourArea(c) >= minimum_size
                and cv2.contourArea(c) <= maximum_size
            ):
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    co = 0
                    for contour1 in cont:
                        co = co + 1
                        dist = cv2.pointPolygonTest(contour1, (cx, cy), True)
                        if dist < 0:
                            if co == le:
                                th = otsu(cr)
                                image0 = cr
                                image00 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
                                ret, thresh0 = cv2.threshold(
                                    image00, th, 255, cv2.THRESH_BINARY
                                )
                                are.append(cv2.contourArea(c))
                                for q in range(0, thresh0.shape[0], 1):
                                    for g in range(0, thresh0.shape[1], 1):
                                        if thresh0[q, g] == 255:
                                            thf0.append(image0[q, g, :])
                                ee = np.array(thf0)
                                thf.append(np.mean(ee, axis=0))
                        else:
                            break
        else:
            if (
                cv2.contourArea(c) >= minimum_size
                and cv2.contourArea(c) <= maximum_size
            ):
                th = otsu(cr)
                image0 = cr
                image00 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
                ret, thresh0 = cv2.threshold(image00, th, 255, cv2.THRESH_BINARY)
                are.append(cv2.contourArea(c))
                for q in range(0, thresh0.shape[0], 1):
                    for g in range(0, thresh0.shape[1], 1):
                        if thresh0[q, g] == 255:
                            thf0.append(image0[q, g, :])
                ee = np.array(thf0)
                thf.append(np.mean(ee, axis=0))

    # show graph
    df = pd.DataFrame(thf)
    df[3] = are
    x = df[3]
    n, bins, patches = plt.hist(x, density=True, bins=50, range=[0, 200])
    fig1, ax1 = plt.subplots()
    ax1.hist(x, density=True, bins=50, range=[0, 200])
    ax1.set_xlabel("size value")
    ax1.set_ylabel("population")
    plt.savefig("size.png")
    plt.close(fig1)

    x = df[0]
    n, bins, patches = plt.hist(x, density=True, bins=25, range=[0, 255])
    # plt.show()
    plt.close()

    # Plot histogram for green value
    x = df[1]
    n, bins, patches = plt.hist(x, density=True, bins=25, range=[0, 255])
    fig2, ax2 = plt.subplots()
    ax2.hist(x, density=True, bins=25, range=[0, 255])
    ax2.set_xlabel("green value")
    ax2.set_ylabel("population")
    # plt.show()
    plt.savefig("green.png")
    plt.close(fig2)

    # find green threshold from histogram
    n = np.diff(n)
    data = n
    peaks, _ = find_peaks(data, height=0.00001)
    valleys, _ = find_peaks(-data, height=-0.00001)

    difference_threshold = 0.01
    first_peak_valley_idx = None

    for peak_idx in peaks:
        next_valleys = valleys[valleys > peak_idx]
        if len(next_valleys) > 0:
            valley_idx = next_valleys[0]
            if abs(data[peak_idx] - data[valley_idx]) > difference_threshold:
                first_peak_valley_idx = valley_idx + 1
                break

    elements_after_first_peak_valley = []
    if first_peak_valley_idx is not None:
        for idx in range(first_peak_valley_idx, len(data)):
            if abs(data[idx]) < 0.0015:
                elements_after_first_peak_valley.append([idx])

        time = np.arange(len(data))
        smoothed_data = n
        plt.figure()
        plt.plot(time, smoothed_data, label="Smoothed Data")
        plt.plot(peaks, data[peaks], "ro", label="Peaks")
        plt.plot(valleys, data[valleys], "go", label="Valleys")
        plt.plot(
            elements_after_first_peak_valley,
            data[elements_after_first_peak_valley],
            "bo",
            label="accepted green",
        )
        plt.xlabel("Time")
        plt.ylabel("Smoothed Data")
        plt.legend()
        plt.savefig("threshold.png")
        plt.close()
        thresh = bins[elements_after_first_peak_valley[0]]
        if thresh[0] <= 40 or thresh[0] >= 150:
            thresh[0] = 80
    else:
        thresh = []
        thresh.append(81)

    # find size threshold from histogram
    n = df[3]
    n, bins, patches = plt.hist(n, density=True, bins=50, range=[0, 200])
    valleys, _ = find_peaks(-n, height=-0.00001)
    n = np.diff(n)
    for i in range(len(n)):
        if i > 0:
            if n[i] < 0 and n[i] > -0.0005:
                break
            if n[i] >= 0:
                break
    if bins[i] <= 10 or bins[i] >= 40:
        bins[i] = 20

    # parameters for last local threshold
    print("minimum green:", thresh[0])
    print("minimum size: ", bins[i])
    minimum_size = bins[i]
    minimum_green = thresh[0]

    return minimum_size,minimum_green ,le
def total_wbc_counter(image_name,image1,minimum_size,maximum_size,le,minimum_green):

    window_size = 15
    sau_threshold = -0.05
    maximum_blue = 100

    # last local threshold to find wbc
    image2 = image1
    image11 = sau(image1, window_size, sau_threshold)
    contours, hierarchy = cv2.findContours(
        image11, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )
    count1 = 0
    countsize2 = 0
    p = 0
    thf = []
    final_contours = []
    for c in contours:
        xx, yy, ww, hh = cv2.boundingRect(c)
        cr = image1[int(yy - p) : int(yy + hh + p), int(xx - p) : int(xx + ww + p)]
        thf0 = []
        if cv2.contourArea(c) >= minimum_size and cv2.contourArea(c) <= maximum_size:
            perimeter = cv2.arcLength(c, True)
            circular = (4 * math.pi * cv2.contourArea(c)) / (perimeter**2)
            count1 += 1
            th = otsu(cr)
            image0 = cr
            image00 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
            ret, thresh0 = cv2.threshold(image00, th, 255, cv2.THRESH_BINARY)
            for q in range(0, thresh0.shape[0], 1):
                for g in range(0, thresh0.shape[1], 1):
                    if thresh0[q, g] == 255:
                        thf0.append(image0[q, g, :])
            ee = np.array(thf0)
            thf.append(np.mean(ee, axis=0))
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                co = 0
                if le != 0:
                    for contour1 in cont:
                        co = co + 1
                        dist = cv2.pointPolygonTest(contour1, (cx, cy), True)
                        if (
                            dist < -8
                            and ((thf[count1 - 1][1])) > minimum_green
                            and (thf[count1 - 1][0]) < maximum_blue
                            and circular > 0
                        ):
                            if co == le:
                                count = count + 1
                                cv2.drawContours(image2, [c], 0, (255, 0, 255), 5)
                                countsize2 = countsize2 + 1
                                final_contours.append(c)
                        else:
                            break
                else:
                    if (
                        ((thf[count1 - 1][1])) > minimum_green
                        and (thf[count1 - 1][0]) < maximum_blue
                        and circular > 0
                    ):
                        cv2.drawContours(image2, [c], 0, (255, 0, 255), 5)
                        final_contours.append(c)
                        countsize2 = countsize2 + 1

    count = 0
    count1 = 0
    countsize = 0
    countsize1 = 0
    countsize2 = 0
    p = 0
    thf = []
    are = []
    image1 = cv2.imread(image_name)
    #image1 = cv2.resize(image1, (4000, 3000))
    for c in final_contours:
        xx, yy, ww, hh = cv2.boundingRect(c)
        cr = image1[int(yy - p) : int(yy + hh + p), int(xx - p) : int(xx + ww + p)]
        count += 1
        thf0 = []
        if cv2.contourArea(c) > minimum_size :
            perimeter = cv2.arcLength(c, True)
            if cv2.contourArea(c) >= minimum_size  and cv2.contourArea(c) <= 2000 :
                count1 += 1
                cv2.imwrite(f"a/{count}.jpg", cr)
                th = otsu(cr)
                image0 = cr
                image00 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
                ret, thresh0 = cv2.threshold(image00, th, 255, cv2.THRESH_BINARY)
                cv2.imwrite(f"a1/{count}.jpg", thresh0)
                are.append(cv2.contourArea(c))

                for q in range(0, thresh0.shape[0], 1):
                    for g in range(0, thresh0.shape[1], 1):
                        if thresh0[q, g] == 255:
                            thf0.append(image0[q, g, :])
                ee = np.array(thf0)
                thf.append(np.mean(ee, axis=0))
    cv2.imwrite("contt2.jpg", image2)
    df_final = pd.DataFrame(thf)
    df_final[3] = df_final[2] / df_final[1]
    df_final[4] = are
    df_final.to_csv("feature.csv")
    return df_final


def feature_extraction(image_path,calibration_coefficient):
    s_area = 6.38
    thickness = 0.022
    calibration_coefficient = calibration_coefficient
    image_name = image_path
    image1 = cv2.imread(image_name)
    #image1 = cv2.resize(image1, (4000, 3000))
    maximum_size = estimation(image1,calibration_coefficient,thickness,s_area)
    cont = noise_recognition(image1,maximum_size)
    minimum_size,minimum_green,le = green_and_size_threshold_finder(image1,cont,maximum_size)
    df_final = total_wbc_counter(image_name,image1,minimum_size,maximum_size,le,minimum_green)
    return df_final
    
    




    