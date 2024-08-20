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
image_name = "7751_crop.jpeg"
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

def kmeans_clustering(df, path_i, n_clusters):
    # Initialize KMeans with variable number of clusters
    kmeans = KMeans(n_clusters=n_clusters, init="random", max_iter=300, n_init=100)
    y_kmeans = kmeans.fit_predict(df.iloc[:, [1, 2]])

    # Count the number of points in each cluster
    cluster_counts = [0] * n_clusters
    for label in y_kmeans:
        cluster_counts[label] += 1
    
    # Filter for relevant file extensions
    included_extensions = ["jpg", "jpeg", "bmp", "png", "gif", "JPG"]
    file_names = [
        fn
        for fn in os.listdir(path_i)
        if any(fn.endswith(ext) for ext in included_extensions)
    ]
    
    # Sort file names naturally
    natsort_file_names = natsorted(file_names)
    
    # Save images to corresponding cluster folders
    for m, file_name in enumerate(natsort_file_names):
        im = cv2.imread(os.path.join(path_i, file_name))
        cluster_label = y_kmeans[m]
        cluster_folder = f"a2/{str(cluster_label+1).zfill(2)}"
        if not os.path.exists(cluster_folder):
            os.makedirs(cluster_folder)
        cv2.imwrite(f"{cluster_folder}/{m+1}_class{cluster_label}.jpg", im)
    
    # Plotting the clusters
    X = df.iloc[:, [2, 1]].values
    colors = ["blue", "red", "green", "yellow", "purple", "orange", "cyan", "magenta", "brown", "pink"]
    
    for i in range(n_clusters):
        plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=10, c=colors[i % len(colors)], label=f"Cluster {i+1}")
    
    plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 0], s=100, c="black", label="Centroids")
    
    plt.title(f"KMeans Clustering with {n_clusters} Clusters")
    plt.xlabel("r")
    plt.ylabel("g")
    plt.legend()
    plt.show()

kmeans_clustering(df, "a", n_clusters=2)
