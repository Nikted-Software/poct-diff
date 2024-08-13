from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from pandas import DataFrame
import pandas as pd

dataset = pd.read_csv("feature.csv")
dataset = dataset.drop(dataset.columns[0], axis=1)
print(dataset)
data = dataset.iloc[:, :].values
x = dataset.iloc[:, [ 3]].values
# print(x.shape)
plt.figure(figsize=(8, 6))
plt.scatter(data[:, [2]], data[:, [1]])
plt.show()
gm = GaussianMixture(
    n_components=2,
    covariance_type="full",
    n_init=100,
    init_params="kmeans",
    max_iter=100,
).fit(x)
gm.get_params()
centers = gm.means_
print(centers)
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 2], data[:, 1], label="data")
#plt.scatter(centers[:, 0], centers[:, 1], c="r", label="centers")
plt.legend()
plt.show()

pred = gm.predict(x)

df = DataFrame({"x": data[:, 2],  "y": data[:, 1], "label": pred})
groups = df.groupby("label")

ig, ax = plt.subplots()
for name, group in groups:
    ax.scatter(group.x, group.y, label=name)

ax.legend()
plt.show()

dataset[5] = pred

print(dataset)

dataset.to_csv('output_with_clusters.csv', index=False)
