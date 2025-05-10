import cv2
from otsu import otsu
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from feature_extraction import feature_extraction
from clustering import kmeans_clustering,spectral_clustering,gaussian_mixture_clustering,agglomerative_clustering,visualize_clusters_on_image

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
image_name = "day3-2m-treat-2-500-700.jpg"
image_name = f"{image_folder}/{image_name}"

image1 = cv2.imread(image_name)
image2 = image1
image3 = image1.copy()
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
dataset = dataset.drop(dataset.columns[0], axis=1)
x = dataset.iloc[:, [2, 1]].values  # r, g

y_kmeans, labeled_dataset = kmeans_clustering(dataset, x, x, base_path, n_clusters=2)
labels_gmm, df_gmm = gaussian_mixture_clustering(df, x, x, base_path, n_clusters=2)
labels_spec, df_spec = spectral_clustering(df, x, x, base_path, n_clusters=2)
labels_agg, df_agg = agglomerative_clustering(df, x, x, base_path, n_clusters=2)

dataset = pd.read_csv("feature.csv")
dataset = dataset.drop(columns=["Unnamed: 0"])

x_centers = dataset["5"].values.astype(int)
y_centers = dataset["6"].values.astype(int)
centers = list(zip(x_centers, y_centers))

x_centers = dataset.iloc[:, 5].values  
y_centers = dataset.iloc[:, 6].values 
centers = list(zip(x_centers.astype(int), y_centers.astype(int)))
visualize_clusters_on_image(image3, centers, labels_gmm, 2)