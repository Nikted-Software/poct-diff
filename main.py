import cv2
from otsu import otsu
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from natsort import natsorted
from threshold_sauvola import sau
from sklearn.cluster import KMeans
import math
from feature_extraction import feature_extraction


import os
import glob

def clear_directory(paths):
    for path in paths:
        files = glob.glob(path)
        for f in files:
            os.remove(f)

def create_directories(paths):
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)

# Define paths
paths_to_clear = [
    "a/*",
    "a1/*",
    "a2/0/*",
    "a2/1/*",
    "a2/2/*",
    "a2/3/*",
    "a2/11/*",
    "a2/22/*",
    "a2/33/*",
    "a2/44/*"
]

paths_to_create = [
    "a",
    "a1",
    "a2/0",
    "a2/1",
    "a2/2",
    "a2/3",
    "a2/11",
    "a2/22",
    "a2/33",
    "a2/44"
]

# Clear files in directories
clear_directory(paths_to_clear)

# Create directories
create_directories(paths_to_create)


image_folder = "data"
image_name = "7751_crop.jpeg"
image_name = f"{image_folder}/{image_name}"
window_size = 15
sau_threshold = -0.05

image1 = cv2.imread(image_name)
image2 = image1
image1[:, :, 0] = 0

df = feature_extraction(image_name,0.93)

x = df[3]
n, bins, patches = plt.hist(x, density=True, bins=50, range=[0, 2])
fig1, ax1 = plt.subplots()
ax1.hist(x, density=True, bins=50, range=[0, 2])
ax1.set_xlabel("size value")
ax1.set_ylabel("population")
plt.savefig("ratio.png")
plt.close(fig1)

m = 0
path_i = "a"
included_extensions = ["jpg", "jpeg", "bmp", "png", "gif", "JPG"]
file_names = [
    fn
    for fn in os.listdir(path_i)
    if any(fn.endswith(ext) for ext in included_extensions)
]
dir_list = file_names
natsort_file_names = natsorted(file_names)

for i in natsort_file_names:
    im = cv2.imread(path_i + "/" + f"{i}")
    if df[2][m] <= 100 :
       cv2.imwrite(f"a2/0/{m+1}.jpg", im)
    else:
       cv2.imwrite(f"a2/1/{m+1}.jpg", im)
    m += 1



kmeans = KMeans(n_clusters=4, init="random", max_iter=300, n_init=100)
y_kmeans = kmeans.fit_predict(df.iloc[:, [1, 2]])
y1 = 0
y2 = 0
y3 = 0
y4 = 0

for i in range(df.shape[0]):
    if y_kmeans[i] == 0:
        y1 += 1
    if y_kmeans[i] == 1:
        y2 += 1
    if y_kmeans[i] == 2:
        y3 += 1
    if y_kmeans[i] == 3:
        y4 += 1
# print(y1, y2, y3, y4)
m = 0
path_i = "a"
relevant_path = path_i
included_extensions = ["jpg", "jpeg", "bmp", "png", "gif", "JPG"]
file_names = [
    fn
    for fn in os.listdir(relevant_path)
    if any(fn.endswith(ext) for ext in included_extensions)
]
dir_list = file_names
natsort_file_names = natsorted(file_names)
for i in natsort_file_names:
    im = cv2.imread(path_i + "/" + f"{i}")
    if y_kmeans[m] == 0:
        cv2.imwrite(f"a2/11/{m+1}class{y_kmeans[m]}.jpg", im)
    if y_kmeans[m] == 1:
        cv2.imwrite(f"a2/22/{m+1}class{y_kmeans[m]}.jpg", im)
    if y_kmeans[m] == 2:
        cv2.imwrite(f"a2/33/{m+1}class{y_kmeans[m]}.jpg", im)
    if y_kmeans[m] == 3:
        cv2.imwrite(f"a2/44/{m+1}class{y_kmeans[m]}.jpg", im)
    m += 1


X = df.iloc[:, [2, 1]].values
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=10, c="blue", label="Cluster 1")
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=10, c="red", label="Cluster 2")
plt.scatter(
    X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=10, c="green", label="Cluster 3"
)
plt.scatter(
    X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=10, c="yellow", label="Cluster 4"
)
plt.scatter(
    kmeans.cluster_centers_[:, 1],
    kmeans.cluster_centers_[:, 0],
    s=100,
    c="black",
    label="Centroids",
)

plt.title("")
plt.xlabel("r")
plt.ylabel("g")
#plt.xticks(range(0, 255,25))
#plt.yticks(range(0,255,25))
plt.legend()
plt.show()
