import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from natsort import natsorted


def visualize_clusters_on_image(image, centers, labels):
    image_copy = image.copy()

    fixed_colors = [
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
    ]

    for idx, (center, label) in enumerate(zip(centers, labels)):
        color = fixed_colors[label % len(fixed_colors)]
        cv2.circle(image_copy, (int(center[0]), int(center[1])), 10, color, -1)

    cv2.imwrite("clustered_image.jpg", image_copy)
    return image_copy


def get_image_files(base_path, extensions=["jpg", "jpeg", "bmp", "png", "gif", "JPG"]):
    file_names = [
        fn
        for fn in os.listdir(base_path)
        if any(fn.endswith(ext) for ext in extensions)
    ]
    return natsorted(file_names)


def save_clustered_images(base_path, file_names, labels, method_name):
    saved_image_filenames = []
    cluster_labels = []
    for m, (file_name, label) in enumerate(zip(file_names, labels)):
        im = cv2.imread(os.path.join(base_path, file_name))
        cluster_folder = os.path.join(
            base_path, method_name, f"{str(label + 1).zfill(2)}"
        )
        if not os.path.exists(cluster_folder):
            os.makedirs(cluster_folder)
        new_file_name = f"{m + 1}_class{label}.jpg"
        saved_image_path = os.path.join(cluster_folder, new_file_name)
        cv2.imwrite(saved_image_path, im)
        saved_image_filenames.append(new_file_name)
        cluster_labels.append(label)


def plot_clusters(df, method_folder, method_name, n_clusters, x_label, y_label):
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


def kmeans_clustering(dataset, x, ax, base_path, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, init="random", max_iter=300, n_init=100)
    y_kmeans = kmeans.fit_predict(x)
    dataset = dataset.copy()
    dataset["label"] = y_kmeans
    file_names = get_image_files(base_path)
    save_clustered_images(base_path, file_names, y_kmeans, "kmeans")
    method_folder = os.path.join(base_path, "kmeans")
    if not os.path.exists(method_folder):
        os.makedirs(method_folder)
    df_plot = pd.DataFrame({"x": ax[:, 0], "y": ax[:, 1], "label": y_kmeans})
    plot_clusters(df_plot, method_folder, "KMeans", n_clusters, "r", "g")
    return y_kmeans, dataset


def gaussian_mixture_clustering(dataset, x, ax, base_path, n_clusters):
    gm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        n_init=100,
        init_params="kmeans",
        max_iter=100,
    )
    pred = gm.fit_predict(x)
    dataset = dataset.copy()
    dataset["label"] = pred
    file_names = get_image_files(base_path)
    save_clustered_images(base_path, file_names, pred, "gaussian_mixture")
    method_folder = os.path.join(base_path, "gaussian_mixture")
    os.makedirs(method_folder, exist_ok=True)
    df_plot = pd.DataFrame({"x": ax[:, 0], "y": ax[:, 1], "label": pred})
    plot_clusters(df_plot, method_folder, "Gaussian Mixture", n_clusters, "x", "y")
    return pred, dataset


def spectral_clustering(dataset, x, ax, base_path, n_clusters):
    clustering = SpectralClustering(
        n_clusters=n_clusters, assign_labels="discretize", random_state=0
    )
    pred = clustering.fit_predict(x)
    dataset = dataset.copy()
    dataset["label"] = pred
    file_names = get_image_files(base_path)
    save_clustered_images(base_path, file_names, pred, "spectral")
    method_folder = os.path.join(base_path, "spectral")
    os.makedirs(method_folder, exist_ok=True)
    df_plot = pd.DataFrame({"x": ax[:, 0], "y": ax[:, 1], "label": pred})
    plot_clusters(df_plot, method_folder, "Spectral", n_clusters, "x", "y")
    return pred, dataset


def agglomerative_clustering(dataset, x, ax, base_path, n_clusters):
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    pred = clustering.fit_predict(x)
    dataset = dataset.copy()
    dataset["label"] = pred
    file_names = get_image_files(base_path)
    save_clustered_images(base_path, file_names, pred, "agglomerative")
    method_folder = os.path.join(base_path, "agglomerative")
    os.makedirs(method_folder, exist_ok=True)
    df_plot = pd.DataFrame({"x": ax[:, 0], "y": ax[:, 1], "label": pred})
    plot_clusters(df_plot, method_folder, "Agglomerative", n_clusters, "r", "g")
    return pred, dataset


def manual_rg_clustering(dataset, x, ax, base_path, threshold):
    pred = (dataset.iloc[:, 3] > threshold).astype(int).values
    dataset = dataset.copy()
    dataset["label"] = pred
    file_names = get_image_files(base_path)
    save_clustered_images(base_path, file_names, pred, "manual_rg")
    method_folder = os.path.join(base_path, "manual_rg")
    os.makedirs(method_folder, exist_ok=True)
    df_plot = pd.DataFrame({"x": ax[:, 0], "y": ax[:, 1], "label": pred})
    plot_clusters(
        df_plot, method_folder, "Manual red to green ratio Clustering", 2, "x", "y"
    )
    return pred, dataset