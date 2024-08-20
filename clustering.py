from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
from pandas import DataFrame
import pandas as pd
from sklearn.cluster import KMeans
import os
from natsort import natsorted
import cv2

def kmeans_clustering(df, path_i, n_clusters):
    
    kmeans = KMeans(n_clusters=n_clusters, init="random", max_iter=300, n_init=100)
    y_kmeans = kmeans.fit_predict(df.iloc[:, [1, 2]])
    cluster_counts = [0] * n_clusters
    for label in y_kmeans:
        cluster_counts[label] += 1
    included_extensions = ["jpg", "jpeg", "bmp", "png", "gif", "JPG"]
    file_names = [
        fn
        for fn in os.listdir(path_i)
        if any(fn.endswith(ext) for ext in included_extensions)
    ]

    natsort_file_names = natsorted(file_names)
    for m, file_name in enumerate(natsort_file_names):
        im = cv2.imread(os.path.join(path_i, file_name))
        cluster_label = y_kmeans[m]
        cluster_folder = f"a2/{str(cluster_label+1).zfill(2)}"
        if not os.path.exists(cluster_folder):
            os.makedirs(cluster_folder)
        cv2.imwrite(f"{cluster_folder}/{m+1}_class{cluster_label}.jpg", im)

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

def gaussian_mixture_clustering(data, n_clusters, output_path="gm.png"):
    
    gm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        n_init=100,
        init_params="kmeans",
        max_iter=100,
    )
    pred = gm.fit_predict(data)
    df = DataFrame({"x": data[:, 2], "y": data[:, 1], "label": pred})
    groups = df.groupby("label")
    fig, ax = plt.subplots()
    for name, group in groups:
        ax.scatter(group.x, group.y, label=name)
    ax.legend()

    plt.savefig(output_path)
    plt.close(fig)

def spectral_clustering(data, x1, n_clusters, output_path="sp.png"):
    
    clustering = SpectralClustering(
        n_clusters=n_clusters,
        assign_labels='discretize',
        random_state=0
    )
    pred = clustering.fit_predict(x1)
    df = DataFrame({"x": data[:, 2], "y": data[:, 1], "label": pred})
    groups = df.groupby("label")
    fig, ax = plt.subplots()
    for name, group in groups:
        ax.scatter(group.x, group.y, label=name)
    ax.legend()
    plt.savefig(output_path)
    plt.close(fig)
    data = DataFrame(data)
    data[5] = pred
    