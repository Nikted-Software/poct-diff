import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from natsort import natsorted

def get_image_files(base_path, extensions=["jpg", "jpeg", "bmp", "png", "gif", "JPG"]):
    """Get sorted list of image files in the directory."""
    file_names = [
        fn for fn in os.listdir(base_path)
        if any(fn.endswith(ext) for ext in extensions)
    ]
    return natsorted(file_names)

def save_clustered_images(base_path, file_names, labels, method_name):
    """Save images into folders based on cluster labels."""
    saved_image_filenames = []
    cluster_labels = []
    for m, (file_name, label) in enumerate(zip(file_names, labels)):
        im = cv2.imread(os.path.join(base_path, file_name))
        cluster_folder = os.path.join(base_path, method_name, f"{str(label + 1).zfill(2)}")
        if not os.path.exists(cluster_folder):
            os.makedirs(cluster_folder)
        new_file_name = f"{m + 1}_class{label}.jpg"
        saved_image_path = os.path.join(cluster_folder, new_file_name)
        cv2.imwrite(saved_image_path, im)
        saved_image_filenames.append(new_file_name)
        cluster_labels.append(label)
    return saved_image_filenames, cluster_labels

def plot_clusters(df, method_folder, method_name, n_clusters, x_label, y_label):
    """Plot and save the clustering results."""
    groups = df.groupby("label")
    fig, ax = plt.subplots()
    for name, group in groups:
        ax.scatter(group.x, group.y, label=f"Cluster {name + 1}")
    ax.legend()
    plt.title(f"{method_name} Clustering with {n_clusters} Clusters")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(os.path.join(method_folder, f"{method_name.lower()}_plot.png"))
    plt.close(fig)

def kmeans_clustering(x, ax, base_path, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, init="random", max_iter=300, n_init=100)
    y_kmeans = kmeans.fit_predict(x)

    method_folder = os.path.join(base_path, "kmeans")
    if not os.path.exists(method_folder):
        os.makedirs(method_folder)

    file_names = get_image_files(base_path)
    save_clustered_images(base_path, file_names, y_kmeans, "kmeans")
    
    df = pd.DataFrame({"x": ax[:, 0], "y": ax[:, 1], "label": y_kmeans})
    plot_clusters(df, method_folder, "KMeans", n_clusters, "r", "g")

def gaussian_mixture_clustering(dff, x, ax, base_path, n_clusters):
    gm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        n_init=100,
        init_params="kmeans",
        max_iter=100,
    )
    pred = gm.fit_predict(x)

    method_folder = os.path.join(base_path, "gaussian_mixture")
    if not os.path.exists(method_folder):
        os.makedirs(method_folder)

    file_names = get_image_files(base_path)
    saved_image_filenames, cluster_labels = save_clustered_images(base_path, file_names, pred, "gaussian_mixture")
    
    dff['saved_file_name'] = saved_image_filenames
    dff['cluster_label'] = cluster_labels
    csv_path = os.path.join(method_folder, "file_labels.csv")
    dff.to_csv(csv_path, index=False)

    df = pd.DataFrame({"x": ax[:, 0], "y": ax[:, 1], "label": pred})
    plot_clusters(df, method_folder, "Gaussian Mixture", n_clusters, "x", "y")

def spectral_clustering(data, x1, base_path, n_clusters):
    clustering = SpectralClustering(
        n_clusters=n_clusters,
        assign_labels='discretize',
        random_state=0
    )
    pred = clustering.fit_predict(x1)

    method_folder = os.path.join(base_path, "spectral")
    if not os.path.exists(method_folder):
        os.makedirs(method_folder)

    file_names = get_image_files(base_path)
    save_clustered_images(base_path, file_names, pred, "spectral")
    
    df = DataFrame({"x": data[:, 2], "y": data[:, 1], "label": pred})
    plot_clusters(df, method_folder, "Spectral", n_clusters, "x", "y")
