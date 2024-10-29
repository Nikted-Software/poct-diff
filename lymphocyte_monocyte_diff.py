import pandas as pd
import numpy as np
from clustering import kmeans_clustering, spectral_clustering, gaussian_mixture_clustering
import matplotlib.pyplot as plt


dataset = pd.read_csv("a/gaussian_mixture/file_labels.csv")
filtered_dataset = dataset[dataset['cluster_label'] == 1]
x_final = filtered_dataset.iloc[:, [3, 4]].values

x_final[:, 0] = x_final[:, 0] / np.max(x_final[:, 0])
x_final[:, 1] = x_final[:, 1] / np.max(x_final[:, 1])

fig1=plt.figure()
plt.scatter(x_final[:, [0]], x_final[:, [1]])
plt.savefig("lym_mon.png")
plt.close(fig1)

gaussian_mixture_clustering(filtered_dataset, x_final, x_final, "a/gaussian_mixture/02", n_clusters=2)
