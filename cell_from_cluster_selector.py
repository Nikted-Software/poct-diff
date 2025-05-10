import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from clustering import gaussian_mixture_clustering
from feature_extraction import feature_extraction
import os

clicking_enabled = True

def on_click(event, ax, img, df, ax_img):
    """Handles the click event to show the corresponding part of the original image with a circle on the cell."""
    global clicking_enabled
    
    if not clicking_enabled:
        return
    
    clicking_enabled = False  # Disable further clicks until the user is ready
    
    print("on_click triggered")
    click_x, click_y = event.xdata, event.ydata
    if click_x is None or click_y is None:
        clicking_enabled = True  # Re-enable clicking
        return
    
    # Find the closest point in the clustered data based on the r and g values
    dist = np.sqrt((df['2'] - click_x) ** 2 + (df['1'] - click_y) ** 2)  
    closest_point_idx = dist.idxmin()  
    
    # Find the corresponding cell in the original image using the index
    cell_position = df.iloc[closest_point_idx][['5', '6']].values  # Columns '5' and '6' are x and y coordinates

    # Assuming the cell positions map directly to the original image's coordinates
    x, y = int(cell_position[0]), int(cell_position[1])

    # Draw a purple circle around the clicked cell on the original image
    img_copy = img.copy()  # Create a copy of the image to avoid modifying the original image

    # Circle parameters
    radius = 10  
    color = (255, 0, 255)  
    thickness = 2  
    
    # Draw the circle on the image
    cv2.circle(img_copy, (x, y), radius, color, thickness)

    # Show the original image with the purple circle drawn around the clicked cell
    ax_img.clear()  # Clear any previous content in the axis
    ax_img.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
    ax_img.set_title(f"Cell at ({x}, {y}) - Highlighted")
    ax_img.axis('off')  # Hide axes
    plt.draw()  # Redraw the plot
    
    ax.clear()
    ax.scatter(df['2'], df['1'], c=df['label'], cmap='viridis', label='Cluster Points')
    ax.legend()
    ax.set_title("Cluster Points with Gaussian Mixture Clustering")
    ax.set_xlabel("r")
    ax.set_ylabel("g")
    plt.draw()  
    
    # Re-enable clicking for the next interaction
    clicking_enabled = True


def plot_clusters(df, method_folder, method_name, n_clusters, x_label, y_label, img, df_gmm):
    """Plots the clusters with their true positions."""
    
    df['label'] = df_gmm['label']
    
    fig, (ax, ax_img) = plt.subplots(1, 2, figsize=(15, 8))  
   
    ax.scatter(df[x_label], df[y_label], c=df['label'], cmap='viridis', label='Cluster Points')

    ax.legend()
    ax.set_title(f"{method_name} Clustering with {n_clusters} Clusters")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Connect the click event to the on_click function
    fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event, ax, img, df, ax_img))

    # Save the plot to the method folder
    os.makedirs(method_folder, exist_ok=True)  # Make sure the directory exists
    plt.savefig(os.path.join(method_folder, f"{method_name.lower()}_plot.png"))
    plt.show()


image_folder = "data"
image_name = "day3-2m-treat-2-500-700.jpg"
image_name = f"{image_folder}/{image_name}"

image3 = cv2.imread(image_name)

df = feature_extraction(image_name, 0.93)
plt.close('all')

dataset = pd.read_csv("feature.csv")
dataset = dataset.drop(dataset.columns[0], axis=1)  



x = dataset.iloc[:, [2, 1]].values  # r, g
x_centers = dataset.iloc[:, 5].values.astype(int)  # Column 5 is 'x'
y_centers = dataset.iloc[:, 6].values.astype(int)  # Column 6 is 'y'

centers = list(zip(x_centers, y_centers))
base_path = "a"
labels_gmm, df_gmm = gaussian_mixture_clustering(df, x, x, base_path, n_clusters=2)
df_gmm = pd.DataFrame({'x': x_centers, 'y': y_centers, 'label': labels_gmm})

plot_clusters(dataset, "output_method", "Gaussian Mixture", 2, "2", "1", image3, df_gmm)
