import os
import cv2
from otsu import otsu

folder_path = 'a/gaussian_mixture/01'
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]
i = 0
are = []
thf0 = []
thf = []
while i < len(image_files):
    file_path = os.path.join(folder_path, image_files[i])
    img = cv2.imread(file_path)
    i += 1
    th = otsu(img)
    image00 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh0 = cv2.threshold(image00, th, 255, cv2.THRESH_BINARY)
    are.append(cv2.contourArea(c))
    for q in range(0, thresh0.shape[0], 1):
        for g in range(0, thresh0.shape[1], 1):
            if thresh0[q, g] == 255:
                thf0.append(image0[q, g, :])
    ee = np.array(thf0)
    thf.append(np.mean(ee, axis=0))
