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

def clear_directory(paths):
    for path in paths:
        for file_or_dir in glob.glob(path, recursive=True):
            if os.path.isdir(file_or_dir):
                for sub_file in glob.glob(os.path.join(file_or_dir, '*'), recursive=True):
                    os.remove(sub_file)
                os.rmdir(file_or_dir)
            else:
                os.remove(file_or_dir)

def create_directories(base_path, n_clusters):

    algorithms = ["kmeans", "gaussian_mixture", "spectral"]
    for algorithm in algorithms:
        for i in range(1, n_clusters + 1):
            cluster_folder = os.path.join(base_path, algorithm, f"{str(i).zfill(2)}")
            if not os.path.isdir(cluster_folder):
                os.makedirs(cluster_folder)

base_path = "result"
n_clusters = 2

paths_to_clear = [
    os.path.join(base_path, "kmeans/*"),
    os.path.join(base_path, "gaussian_mixture/*"),
    os.path.join(base_path, "spectral/*")
]

clear_directory(paths_to_clear)
create_directories(base_path, n_clusters)

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
path_i = base_path
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

kmeans_clustering(df, base_path, n_clusters=2)
gaussian_mixture_clustering(data,x, base_path, n_clusters=n_clusters)
updated_data = spectral_clustering(data, x1, base_path ,n_clusters=n_clusters)
