from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
from pandas import DataFrame
import pandas as pd

#params
n_clusters = 2
dataset = pd.read_csv("feature.csv")

#preprocessing
dataset = dataset.drop(dataset.columns[0], axis=1)
data = dataset.iloc[:, :].values
x = dataset.iloc[:, [1,2]].values
x1 = dataset.iloc[:, [1,2]].values
x1[:,[0,1]] = x1[:,[0,1]]/255

#plot data
fig1=plt.figure()
plt.scatter(data[:, [2]], data[:, [1]])
plt.savefig("data.png")
plt.close(fig1)


gm = GaussianMixture(
    n_components=n_clusters,
    covariance_type="full",
    n_init=100,
    init_params="kmeans",
    max_iter=100,
)

pred = gm.fit_predict(x)
df = DataFrame({"x": data[:, 2],  "y": data[:, 1], "label": pred})
groups = df.groupby("label")
fig2, ax = plt.subplots()
for name, group in groups:
    ax.scatter(group.x, group.y, label=name)
ax.legend()
plt.savefig("gm.png")
plt.close(fig2)

clustering = SpectralClustering(
        n_clusters=n_clusters,
        assign_labels='discretize',
        random_state=0)
pred =  clustering.fit_predict(x1)
df = DataFrame({"x": data[:, 2],  "y": data[:, 1], "label": pred})
groups = df.groupby("label")
fig3, ax = plt.subplots()
for name, group in groups:
    ax.scatter(group.x, group.y, label=name)
ax.legend()
plt.savefig("sp.png")
plt.close(fig3)

dataset[5] = pred

#print(dataset)
print(sum(pred),len(pred))
dataset.to_csv('output_with_clusters.csv', index=False)
