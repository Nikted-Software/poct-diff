import cv2
from otsu import otsu
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from natsort import natsorted
from threshold_sauvola import sau
import math
from feature_extraction import feature_extraction
from clustering import kmeans_clustering,spectral_clustering,gaussian_mixture_clustering

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

clear_directory(paths_to_clear)
create_directories(paths_to_create)

image_folder = "data"
image_name = "7791_crop.jpeg"
image_name = f"{image_folder}/{image_name}"

image1 = cv2.imread(image_name)
image2 = image1
image1[:, :, 0] = 0

df = feature_extraction(image_name,0.93)

x = df[3]
n, bins, patches = plt.hist(x, density=True, bins=50, range=[0, 3])
fig1, ax1 = plt.subplots()
ax1.hist(x, density=True, bins=50, range=[0, 3])
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

#params
n_clusters = 2
dataset = pd.read_csv("feature.csv")

#preprocessing
dataset = dataset.drop(dataset.columns[0], axis=1)
data = dataset.iloc[:, :].values
x = dataset.iloc[:, [1,2]].values
x1 = dataset.iloc[:, [1,2]].values
x1[:,[0,1]] = x1[:,[0,1]]/255

#plot data
fig1=plt.figure()
plt.scatter(data[:, [2]], data[:, [1]])
plt.savefig("data.png")
plt.close(fig1)

kmeans_clustering(df, "a", n_clusters=2)
gaussian_mixture_clustering(data, n_clusters=n_clusters, output_path="gm.png")
updated_data = spectral_clustering(data, x1, n_clusters=n_clusters, output_path="sp.png")
