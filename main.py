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
from clustering import kmeans_clustering,spectral_clustering,gaussian_mixture_clustering,agglomerative_clustering

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

    algorithms = ["kmeans", "gaussian_mixture", "spectral","agglomerative"]
    for algorithm in algorithms:
        for i in range(1, n_clusters + 1):
            cluster_folder = os.path.join(base_path, algorithm, f"{str(i).zfill(2)}")
            if not os.path.isdir(cluster_folder):
                os.makedirs(cluster_folder)

base_path = "a"
n_clusters = 2

paths_to_clear = [
    os.path.join(base_path, "kmeans/*"),
    os.path.join(base_path, "gaussian_mixture/*"),
    os.path.join(base_path, "spectral/*"),
    os.path.join(base_path, "agglomerative/*"),
    "a/*"
]

clear_directory(paths_to_clear)
create_directories(base_path, n_clusters)

files = glob.glob("a1/*")
for f in files:
    os.remove(f)

mypath = "a1"
if not os.path.isdir(mypath):
    os.makedirs(mypath)


image_folder = "data"
image_name = "7926.jpeg"
image_name = f"{image_folder}/{image_name}"

image1 = cv2.imread(image_name)
image2 = image1
image1[:, :, 0] = 0

df = feature_extraction(image_name,0.93)

x = df[3]
n, bins, patches = plt.hist(x, density=True, bins=50, range=[0, 3])
fig1, ax1 = plt.subplots()
ax1.hist(x, density=True, bins=50, range=[0, 3])
ax1.set_xlabel("r/g")
ax1.set_ylabel("population")
plt.savefig("ratio.png")
plt.close(fig1)

dataset = pd.read_csv("feature.csv")

#preprocessing
dataset = dataset.drop(dataset.columns[0], axis=1)
x1 = dataset.iloc[:, [3]].values
x = dataset.iloc[:, [2,1]].values

#x1 = dataset.iloc[:, [1,2]].values
#x2 = dataset.iloc[:, [4]].values
#x1[:,[0]] = np.multiply(x1[:,[0]],x2)/np.max(np.multiply(x1[:,[0]],x2))
#x1[:,[1]] = np.multiply(x1[:,[1]],x2)/np.max(np.multiply(x1[:,[1]],x2))
#
##plot data
#fig1=plt.figure()
#plt.scatter(x[:, [0]], x[:, [1]])
#plt.savefig("data.png")
#plt.close(fig1)
#
#fig1=plt.figure()
#plt.scatter(x1[:, [0]], x1[:, [1]])
#plt.savefig("data1.png")
#plt.close(fig1)


kmeans_clustering(x1,x, base_path, n_clusters=n_clusters)
gaussian_mixture_clustering(df,x1,x, base_path, n_clusters=n_clusters)
spectral_clustering(x1,x, base_path ,n_clusters=n_clusters)
agglomerative_clustering(x1,x, base_path, n_clusters=n_clusters)
