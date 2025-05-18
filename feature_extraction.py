import cv2
from scipy.signal import find_peaks
from otsu import otsu
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# matplotlib.use('Agg')
import warnings
from threshold_sauvola import sau
import math
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

warnings.filterwarnings("ignore", message="invalid value encountered in divide")


###############################################################
# local threshold with small window to find order of wbc number
###############################################################
def estimation(image1, calibration_coefficient, thickness, s_area):
    window_size = 15
    sau_threshold = -0.05
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


###############################################################
# local threshold with big window to recognize stains and noises
###############################################################
def noise_recognition(image1, maximum_size):

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

    ###############################################################
    # Determine whether there is a channel in the image or not
    ###############################################################
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


###############################################################
# local threshold with small window to find size and green threshold
###############################################################
def green_and_size_threshold_finder(image1, cont, maximum_size):

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

    ###############################################################
    # Plot histogram for red value
    ###############################################################
    x = df[2]
    n, bins, patches = plt.hist(x, density=True, bins=25, range=[0, 255])
    fig2, ax2 = plt.subplots()
    ax2.hist(x, density=True, bins=25, range=[0, 255])
    ax2.set_xlabel("red value")
    ax2.set_ylabel("population")
    # plt.show()
    plt.savefig("red.png")
    plt.close(fig2)

    ###############################################################
    # find red threshold from histogram
    ###############################################################
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
        plt.savefig("red_threshold.png")
        plt.close()
        thresh_red = bins[elements_after_first_peak_valley[0]]
        if thresh_red[0] <= 40 or thresh_red[0] >= 150:
            thresh_red[0] = 80
    else:
        thresh_red = []
        thresh_red.append(81)

    ###############################################################
    # Plot histogram for green value
    ###############################################################
    x = df[1]
    n, bins, patches = plt.hist(x, density=True, bins=25, range=[0, 255])
    fig2, ax2 = plt.subplots()
    ax2.hist(x, density=True, bins=25, range=[0, 255])
    ax2.set_xlabel("green value")
    ax2.set_ylabel("population")
    # plt.show()
    plt.savefig("green.png")
    plt.close(fig2)

    ###############################################################
    # find green threshold from histogram
    ###############################################################
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
        plt.savefig("green_threshold.png")
        plt.close()
        thresh = bins[elements_after_first_peak_valley[0]]
        if thresh[0] <= 40 or thresh[0] >= 150:
            thresh[0] = 80
    else:
        thresh = []
        thresh.append(81)

    ###############################################################
    # find size threshold from histogram
    ###############################################################
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
    print(bins[i])
    if bins[i] <= 5 or bins[i] >= 40:
        bins[i] = 20

    print("minimum green:", thresh[0])
    print("minimum size: ", bins[i])
    print("minimum red: ", thresh_red[0])
    minimum_size = bins[i]
    minimum_green = thresh[0]
    minimum_red = thresh_red[0]
    return minimum_size, minimum_green, minimum_red, le


###############################################################
# last local threshold to find wbc count
###############################################################
def total_wbc_counter(
    image_name, image1, minimum_size, maximum_size, le, minimum_green, cont, minimum_red
):

    window_size = 15
    sau_threshold = -0.05
    maximum_blue = 100
    image2 = image1
    image11 = sau(image1, window_size, sau_threshold)
    contours, hierarchy = cv2.findContours(
        image11, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )
    count = 0
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
                            and (thf[count1 - 1][0]) < maximum_blue
                            and circular > 0
                            and (
                                ((thf[count1 - 1][1])) > minimum_green
                                or ((thf[count1 - 1][2])) > minimum_red
                            )
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
                        (thf[count1 - 1][0]) < maximum_blue
                        and circular > 0
                        and (
                            ((thf[count1 - 1][1])) > minimum_green
                            or ((thf[count1 - 1][2])) > minimum_red
                        )
                    ):
                        cv2.drawContours(image2, [c], 0, (255, 0, 255), 5)
                        final_contours.append(c)
                        countsize2 = countsize2 + 1

    count = 0
    count1 = 0
    countsize2 = 0
    p = 0
    thf = []
    are = []
    centers = []
    image1 = cv2.imread(image_name)
    # image1 = cv2.resize(image1, (4000, 3000))
    for c in final_contours:
        xx, yy, ww, hh = cv2.boundingRect(c)
        cr = image1[int(yy - p) : int(yy + hh + p), int(xx - p) : int(xx + ww + p)]
        count += 1
        thf0 = []
        if cv2.contourArea(c) > minimum_size:
            perimeter = cv2.arcLength(c, True)
            if cv2.contourArea(c) >= minimum_size and cv2.contourArea(c) <= 2000:
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

                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = 0, 0
                centers.append((cx, cy))

    m, n = 30, 30 
    patch_size = 3
    half_patch = patch_size // 2
    min_distance = 20

    h, w = image1.shape[:2]
    cell_h = h // m
    cell_w = w // n
    
    mask = np.zeros(image1.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, final_contours, -1, color=255, thickness=cv2.FILLED)
    background_mask = cv2.bitwise_not(mask)
    dist_transform = cv2.distanceTransform(background_mask, cv2.DIST_L2, 5)

    # Step 4: Loop over each grid cell and sample its center if valid
    sampled_coords = []

    for i in range(m):
        for j in range(n):
            y = i * cell_h + cell_h // 2
            x = j * cell_w + cell_w // 2
            if (
                y - half_patch < 0 or y + half_patch >= h or
                x - half_patch < 0 or x + half_patch >= w
            ):
                continue

            if background_mask[y, x] == 255 and dist_transform[y, x] >= min_distance:
                sampled_coords.append((y, x))

    # Step 5: Extract BGR patch mean for valid sampled points
    patch_means = []
    xy_coords = []

    for y, x in sampled_coords:
        patch = image1[
            y - half_patch : y + half_patch + 1,
            x - half_patch : x + half_patch + 1
        ]
        patch_mean = patch.reshape(-1, 3).mean(axis=0)
        patch_means.append(patch_mean)
        xy_coords.append((x, y))  

    df_background = pd.DataFrame(patch_means, columns=["B", "G", "R"])
    df_background["x"] = [pt[0] for pt in xy_coords]
    df_background["y"] = [pt[1] for pt in xy_coords]
    df_background["intensity"] = (
        df_background["B"] + df_background["G"] + df_background["R"]
    ) / 3
    df_background.to_csv("background_samples_patch.csv", index=False)

    mean_intensity = df_background["intensity"].mean()
    variance_intensity = df_background["intensity"].var()
    std_intensity = np.sqrt(variance_intensity)
    print(f"Mean intensity: {mean_intensity:.2f}")
    print(f"Standard deviation: {std_intensity:.2f}")

    x = df_background["x"].values
    y = df_background["y"].values
    z = df_background["intensity"].values
    x_vals = df_background["x"].values
    y_vals = df_background["y"].values
    z_vals = df_background["intensity"].values

    image_with_points = image1.copy()
    for y, x in sampled_coords:
        cv2.drawMarker(
            image_with_points,
            (x, y),
            color=(0, 0, 255),
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=10,
            thickness=3,
        )
    image_rgb = cv2.cvtColor(image_with_points, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 8))
    plt.imshow(image_rgb)
    plt.title("Sampled Background Points (+) on Original Image")
    plt.axis("off")
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x_vals, y_vals, z_vals, color="red", s=20, label="Sampled Points")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Total Intensity (B+G+R)")
    ax.set_zlim(0, 255)
    ax.legend()
    plt.title("3D Scatter of Sampled Background Intensities")
    plt.tight_layout()
    plt.show(block=False)
    plt.show()

    avg_x = df_background.groupby("x")["intensity"].mean().reset_index()
    plt.figure(figsize=(8, 4))
    plt.plot(avg_x["x"], avg_x["intensity"], color="blue")
    plt.xlabel("X Coordinate")
    plt.ylabel("Average Intensity")
    plt.title("Average Intensity vs X Axis")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    avg_y = df_background.groupby("y")["intensity"].mean().reset_index()
    plt.figure(figsize=(8, 4))
    plt.plot(avg_y["y"], avg_y["intensity"], color="green")
    plt.xlabel("Y Coordinate")
    plt.ylabel("Average Intensity")
    plt.title("Average Intensity vs Y Axis")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    poly = PolynomialFeatures(degree=6)
    X_poly = poly.fit_transform(np.column_stack((x_vals, y_vals)))
    model = LinearRegression().fit(X_poly, z_vals)
    x_range = np.linspace(x_vals.min(), x_vals.max(), 100)
    y_range = np.linspace(y_vals.min(), y_vals.max(), 100)
    grid_x, grid_y = np.meshgrid(x_range, y_range)
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    grid_poly = poly.transform(grid_points)
    grid_z = model.predict(grid_poly).reshape(grid_x.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(grid_x, grid_y, grid_z, cmap="viridis", alpha=0.8)
    ax.scatter(x_vals, y_vals, z_vals, color="red", s=20, label="Sampled Points")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Total Intensity (B+G+R)")
    ax.set_zlim(0, 255)
    ax.legend()
    plt.title("3D Regression Surface with Sampled Background Intensities")
    plt.tight_layout()
    plt.show()

    # Prepare the regression grid
    h, w = image1.shape[:2]
    x_full, y_full = np.meshgrid(np.arange(w), np.arange(h))
    flat_xy = np.column_stack((x_full.ravel(), y_full.ravel()))
    degree = 6
    poly = PolynomialFeatures(degree=degree)
    X_sample = poly.fit_transform(np.column_stack((x_vals, y_vals)))
    background_surface = np.zeros_like(image1, dtype=np.float32)
    for i, color in enumerate(["B", "G", "R"]):
        z = df_background[color].values
        model = LinearRegression().fit(X_sample, z)
        X_full_poly = poly.transform(flat_xy)
        predicted = model.predict(X_full_poly).reshape((h, w))

        background_surface[:, :, i] = predicted
    image_float = image1.astype(np.float32)
    image_subtracted = image_float - background_surface
    image_subtracted = np.clip(image_subtracted, 0, 255).astype(np.uint8)
    cv2.imwrite("image_background_surface_subtracted.jpg", image_subtracted)

    mean_bgr = df_background[["B", "G", "R"]].mean().values
    print(f"Mean background color (BGR): {mean_bgr}")
    image_subtracted_mean = image1.astype(np.float32) - mean_bgr
    image_subtracted_mean = np.clip(image_subtracted_mean, 0, 255).astype(np.uint8)
    cv2.imwrite("image_background_mean_subtracted.jpg", image_subtracted_mean)

    channel_names = ["Blue", "Green", "Red"]
    channel_colors = ["blue", "green", "red"]
    for i, (name, color) in enumerate(zip(channel_names, channel_colors)):
        surface = background_surface[:, :, i]
        step = max(1, h // 100)
        x_surf = x_full[::step, ::step]
        y_surf = y_full[::step, ::step]
        z_surf = surface[::step, ::step]
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(x_surf, y_surf, z_surf, cmap="viridis", alpha=0.8)
        ax.scatter(
            x_vals,
            y_vals,
            df_background[color[0].upper()],
            color=color,
            s=10,
            label=f"Sampled {name} points",
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel(f"{name} Value")
        ax.set_zlim(0, 255)
        ax.legend()
        plt.title(f"{name} Channel Regression Surface with Sampled Points")
        plt.tight_layout()
        plt.show()

    cv2.imwrite("contt2.jpg", image2)
    df_final = pd.DataFrame(thf)

    mean_intensity = ((df_final[0] + df_final[1] + df_final[2]) / 3).mean()
    std_intensity = ((df_final[0] + df_final[1] + df_final[2]) / 3).std()
    print(f"Mean cell intensity: {mean_intensity:.2f}")
    print(f"Standard deviation of cell intensity: {std_intensity:.2f}")

    df_final[3] = df_final[2] / df_final[1]
    df_final[4] = are
    df_final[5] = [pt[0] for pt in centers]
    df_final[6] = [pt[1] for pt in centers]
    df_final.to_csv("feature.csv")
    print("cell number:", count1)

    # Convert x and y to integers (they must be pixel indices)
    x_cell = df_final[5].astype(int).clip(0, image1.shape[1] - 1)
    y_cell = df_final[6].astype(int).clip(0, image1.shape[0] - 1)
    df_final[0] = df_final[0] - background_surface[y_cell, x_cell, 0]
    df_final[1] = df_final[1] - background_surface[y_cell, x_cell, 1]
    df_final[2] = df_final[2] - background_surface[y_cell, x_cell, 2]
    df_final[3] = df_final[2] / df_final[1]
    df_final[4] = are
    df_final[5] = [pt[0] for pt in centers]
    df_final[6] = [pt[1] for pt in centers]
    df_final.to_csv("feature_background_subtracted.csv")


def feature_extraction(image_name, calibration_coefficient):
    s_area = 6.38
    thickness = 0.022
    image1 = cv2.imread(image_name)
    # image1 = cv2.resize(image1, (4000, 3000))
    maximum_size = estimation(image1, calibration_coefficient, thickness, s_area)
    cont = noise_recognition(image1, maximum_size)
    minimum_size, minimum_green, minimum_red, le = green_and_size_threshold_finder(
        image1, cont, maximum_size
    )
    total_wbc_counter(
        image_name,
        image1,
        minimum_size,
        maximum_size,
        le,
        minimum_green,
        cont,
        minimum_red,
    )
