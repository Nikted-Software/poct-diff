import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from clustering import gaussian_mixture_clustering
from feature_extraction import feature_extraction
import os

clicking_enabled = True

def on_click(event, ax, img, df, ax_img):
    global clicking_enabled

    if not clicking_enabled:
        return

    clicking_enabled = False

    click_x, click_y = event.xdata, event.ydata
    if click_x is None or click_y is None:
        clicking_enabled = True
        return

    dist = np.sqrt((df['2'] - click_x) ** 2 + (df['1'] - click_y) ** 2)
    closest_point_idx = dist.idxmin()
    cell_position = df.iloc[closest_point_idx][['5', '6']].values

    x, y = int(cell_position[0]), int(cell_position[1])

    img_copy = img.copy()
    cv2.circle(img_copy, (x, y), 10, (255, 0, 255), 2)

    ax_img.clear()
    ax_img.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
    ax_img.set_title(f"Cell at ({x}, {y}) - Highlighted")
    ax_img.axis('off')
    plt.draw()

    ax.clear()
    ax.scatter(df['2'], df['1'], c=df['label'], cmap='viridis', label='Cluster Points')
    ax.legend()
    ax.set_title("Cluster Points with Gaussian Mixture Clustering")
    ax.set_xlabel("r")
    ax.set_ylabel("g")
    plt.draw()

    clicking_enabled = True

def plot_clusters(df, method_folder, method_name, n_clusters, x_label, y_label, img, df_gmm):
    df['label'] = df_gmm['label']

    fig, (ax, ax_img) = plt.subplots(1, 2, figsize=(15, 8))
    ax.scatter(df[x_label], df[y_label], c=df['label'], cmap='viridis', label='Cluster Points')
    ax.legend()
    ax.set_title(f"{method_name} Clustering with {n_clusters} Clusters")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event, ax, img, df, ax_img))

    # === Manual selection with Lasso ===
    selector = None  # <-- store the selector

    def on_select(verts):
        path = Path(verts)
        coords = np.column_stack((df[x_label], df[y_label]))
        selected = path.contains_points(coords)
        selected_count = np.sum(selected)
        print(f"Manually selected points: {selected_count}")

        # Highlight selected points
        ax.scatter(df[x_label][selected], df[y_label][selected],
                   edgecolor='red', facecolor='none', s=100, label='Selected')
        ax.legend()
        plt.draw()

        # Highlight on original image
        selected_points = df[selected]
        img_copy = img.copy()
        for _, row in selected_points.iterrows():
            x, y = int(row['5']), int(row['6'])
            cv2.circle(img_copy, (x, y), 10, (255, 0, 255), 2)

        ax_img.clear()
        ax_img.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
        ax_img.set_title(f"{selected_count} Points Highlighted in Image")
        ax_img.axis('off')
        plt.draw()

        if os.path.exists("selected_points.csv"):
            os.remove("selected_points.csv")
        selected_points.to_csv("selected_points.csv", index=False)
        print("Saved selected points to selected_points.csv")

    # Keep reference to the LassoSelector to prevent garbage collection
    selector = LassoSelector(ax, on_select)
    # ==================================

    os.makedirs(method_folder, exist_ok=True)
    plt.savefig(os.path.join(method_folder, f"{method_name.lower()}_plot.png"))
    plt.show()


image_folder = "data"
image_name = "2m_treat2.jpg"
image_name = f"{image_folder}/{image_name}"

image3 = cv2.imread(image_name)

df = feature_extraction(image_name, 0.93)
plt.close('all')

dataset = pd.read_csv("feature.csv")
dataset = dataset.drop(dataset.columns[0], axis=1)

x = dataset.iloc[:, [2, 1]].values
x_centers = dataset.iloc[:, 5].values.astype(int)
y_centers = dataset.iloc[:, 6].values.astype(int)

centers = list(zip(x_centers, y_centers))
base_path = "a"
labels_gmm, df_gmm = gaussian_mixture_clustering(df, x, x, base_path, n_clusters=2)
df_gmm = pd.DataFrame({'x': x_centers, 'y': y_centers, 'label': labels_gmm})

plot_clusters(dataset, "output_method", "Gaussian Mixture", 2, "2", "1", image3, df_gmm)
