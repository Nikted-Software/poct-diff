import os
import cv2
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from natsort import natsorted
import pandas as pd

def kmeans_clustering(x,ax, base_path, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, init="random", max_iter=300, n_init=100)
    y_kmeans = kmeans.fit_predict(x)
    
    included_extensions = ["jpg", "jpeg", "bmp", "png", "gif", "JPG"]
    file_names = [
        fn for fn in os.listdir(base_path)
        if any(fn.endswith(ext) for ext in included_extensions)
    ]

    natsort_file_names = natsorted(file_names)

    # Check for consistency between number of images and number of predicted clusters
    min_len = min(len(natsort_file_names), len(y_kmeans))

    for m in range(min_len):
        file_name = natsort_file_names[m]
        im = cv2.imread(os.path.join(base_path, file_name))
        cluster_label = y_kmeans[m]
        cluster_folder = os.path.join(base_path, "kmeans", f"{str(cluster_label+1).zfill(2)}")
        if not os.path.exists(cluster_folder):
            os.makedirs(cluster_folder)
        cv2.imwrite(f"{cluster_folder}/{m+1}_class{cluster_label}.jpg", im)

    
    X = ax
    colors = ["blue", "red", "green", "yellow", "purple", "orange", "cyan", "magenta", "brown", "pink"]
    for i in range(n_clusters):
        plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=10, c=colors[i % len(colors)], label=f"Cluster {i+1}")
    plt.title(f"KMeans Clustering with {n_clusters} Clusters")
    plt.xlabel("r")
    plt.ylabel("g")
    plt.legend()
    plt.savefig(os.path.join(base_path, "kmeans", "kmeans_plot.png"))
    plt.close()


def gaussian_mixture_clustering(dff, x, ax, base_path, n_clusters):
    gm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        n_init=100,
        init_params="kmeans",
        max_iter=100,
    )
    pred = gm.fit_predict(x)
    df = pd.DataFrame({"x": ax[:, 0], "y": ax[:, 1], "label": pred})

    included_extensions = ["jpg", "jpeg", "bmp", "png", "gif", "JPG"]
    file_names = [
        fn for fn in os.listdir(base_path)
        if any(fn.endswith(ext) for ext in included_extensions)
    ]
    natsort_file_names = natsorted(file_names)
    min_len = min(len(natsort_file_names), len(pred))
    saved_image_filenames = []
    cluster_labels = []

    for m in range(min_len):
        file_name = natsort_file_names[m]
        im = cv2.imread(os.path.join(base_path, file_name))
        cluster_label = pred[m]
        cluster_folder = os.path.join(base_path, "gaussian_mixture", f"{str(cluster_label + 1).zfill(2)}")
        if not os.path.exists(cluster_folder):
            os.makedirs(cluster_folder)
        new_file_name = f"{m + 1}_class{cluster_label}.jpg"
        saved_image_path = os.path.join(cluster_folder, new_file_name)
        cv2.imwrite(saved_image_path, im)
        saved_image_filenames.append(new_file_name)
        cluster_labels.append(cluster_label)
    dff['saved_file_name'] = saved_image_filenames
    dff['cluster_label'] = cluster_labels
    csv_path = os.path.join(base_path, "gaussian_mixture", "file_labels.csv")
    dff.to_csv(csv_path, index=False)
    groups = df.groupby("label")
    fig, ax = plt.subplots()
    for name, group in groups:
        ax.scatter(group.x, group.y, label=name)
    ax.legend()
    plt.title(f"Gaussian Mixture Clustering with {n_clusters} Clusters")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(os.path.join(base_path, "gaussian_mixture", "gaussian_mixture_plot.png"))
    plt.close(fig)
    
def spectral_clustering(data, x1, base_path, n_clusters):
    clustering = SpectralClustering(
        n_clusters=n_clusters,
        assign_labels='discretize',
        random_state=0
    )
    pred = clustering.fit_predict(x1)

    df = DataFrame({"x": data[:, 2], "y": data[:, 1], "label": pred})
    included_extensions = ["jpg", "jpeg", "bmp", "png", "gif", "JPG"]
    file_names = [
        fn for fn in os.listdir(base_path)
        if any(fn.endswith(ext) for ext in included_extensions)
    ]
    
    natsort_file_names = natsorted(file_names)

    # Check for consistency between number of images and number of predicted clusters
    min_len = min(len(natsort_file_names), len(pred))

    for m in range(min_len):
        file_name = natsort_file_names[m]
        im = cv2.imread(os.path.join(base_path, file_name))
        cluster_label = pred[m]
        cluster_folder = os.path.join(base_path, "spectral", f"{str(cluster_label+1).zfill(2)}")
        if not os.path.exists(cluster_folder):
            os.makedirs(cluster_folder)
        cv2.imwrite(f"{cluster_folder}/{m+1}_class{cluster_label}.jpg", im)
    
    groups = df.groupby("label")
    fig, ax = plt.subplots()
    for name, group in groups:
        ax.scatter(group.x, group.y, label=name)
    ax.legend()
    plt.title(f"Spectral Clustering with {n_clusters} Clusters")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(os.path.join(base_path, "spectral", "spectral_clustering_plot.png"))
    plt.close(fig)
