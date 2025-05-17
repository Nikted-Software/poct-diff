import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
from feature_extraction import feature_extraction
from clustering import kmeans_clustering,spectral_clustering,gaussian_mixture_clustering,agglomerative_clustering,visualize_clusters_on_image,manual_rg_clustering

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

    algorithms = ["kmeans", "gaussian_mixture", "spectral","agglomerative", "manual_rg"]
    for algorithm in algorithms:
        for i in range(1, n_clusters + 1):
            cluster_folder = os.path.join(base_path, algorithm, f"{str(i).zfill(2)}")
            if not os.path.isdir(cluster_folder):
                os.makedirs(cluster_folder)

def plot_histogram(data, bins, data_range, xlabel, ylabel, xticks=None, filename="output.png"):
    fig, ax = plt.subplots()
    ax.hist(data, density=True, bins=bins, range=data_range)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xticks is not None:
        ax.set_xticks(xticks)
    plt.savefig(filename)
    plt.close(fig)

image_folder = "data/14040224"
image_name = "20m_treat_2ao_3_crop.jpg"
base_path = "a"
n_clusters = 2

paths_to_clear = [
    os.path.join(base_path, "kmeans/*"),
    os.path.join(base_path, "gaussian_mixture/*"),
    os.path.join(base_path, "spectral/*"),
    os.path.join(base_path, "agglomerative/*"),
    os.path.join(base_path, "manual_rg/*"),
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

image_name = f"{image_folder}/{image_name}"
image1 = cv2.imread(image_name)
image2 = image1
image3 = image1.copy()
image1[:, :, 0] = 0


df = feature_extraction(image_name,0.93)

plot_histogram(df[1], bins=20, data_range=[0, 255], xlabel="green", ylabel="population", filename="green_cell.png")
plot_histogram(df[2], bins=20, data_range=[0, 255], xlabel="red", ylabel="population", filename="red_cell.png")
plot_histogram(df[3], bins=20, data_range=[0, 3], xlabel="r/g", ylabel="population", xticks=np.arange(0, 3.1, 0.2), filename="red_to_green_ratio.png")

dataset = pd.read_csv("feature.csv")
dataset = dataset.drop(dataset.columns[0], axis=1)

# Column 0: Blue channel (B)
# Column 1: Green channel (G)
# Column 2: Red channel (R)
# Column 3: Red/Green ratio (R/G)
# Column 4: Area
# Column 5: Center X coordinate
# Column 6: Center Y coordinate
x = dataset.iloc[:, [2, 1]].values  

y_kmeans, labeled_dataset = kmeans_clustering(dataset, x, x, base_path, n_clusters=2)
labels_gmm, df_gmm = gaussian_mixture_clustering(df, x, x, base_path, n_clusters=2)
labels_spec, df_spec = spectral_clustering(df, x, x, base_path, n_clusters=2)
labels_agg, df_agg = agglomerative_clustering(df, x, x, base_path, n_clusters=2)
labels_manual, df_manual = manual_rg_clustering(df, x, x, base_path, 1.2)

dataset = pd.read_csv("feature.csv")
dataset = dataset.drop(columns=["Unnamed: 0"])

x_centers = dataset["5"].values.astype(int)
y_centers = dataset["6"].values.astype(int)
centers = list(zip(x_centers, y_centers))

x_centers = dataset.iloc[:, 5].values  
y_centers = dataset.iloc[:, 6].values 
centers = list(zip(x_centers.astype(int), y_centers.astype(int)))
visualize_clusters_on_image(image3, centers, labels_manual)