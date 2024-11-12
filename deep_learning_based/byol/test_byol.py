import torch
from torch import nn
from lightly.data import LightlyDataset
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import shutil
import matplotlib.pyplot as plt
import os
from transform import SimCLRTransform
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad
import copy


random_number = 43
torch.manual_seed(random_number)
torch.cuda.manual_seed(random_number)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.encoder_conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.encoder_relu1 = nn.ReLU()
        self.encoder_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_conv2 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1)
        self.encoder_relu2 = nn.ReLU()
        self.encoder_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.encoder_conv1(x)
        x = self.encoder_relu1(x)
        x = self.encoder_pool1(x)
        x = self.encoder_conv2(x)
        x = self.encoder_relu2(x)
        x = self.encoder_pool2(x)
        return x

class BYOL(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = BYOLProjectionHead(128, 64, 16)
        self.prediction_head = BYOLPredictionHead(16, 32, 16)

        # Momentum encoder
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p

    def forward_momentum(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        return z.detach()

model = BYOL(Backbone())

model.load_state_dict(torch.load('best_byol_model.pth'))

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

transform = SimCLRTransform(input_size=16,
        cj_prob=0,
        random_gray_scale=0,
        gaussian_blur=0,
        vf_prob=0,
        hf_prob=0,
        rr_prob=0,
        )

dataset = LightlyDataset("../2", transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

features = []
image_filenames = []

print("Starting feature extraction")
with torch.no_grad():
    for (view0, view1), _, batch_filenames in dataloader:
        view0 = view0.to(device)
        z = model.forward_momentum(view0)  # Extract features using the momentum encoder
        features.append(z.flatten(start_dim=1).cpu())  # Collect features for clustering
        image_filenames.extend(batch_filenames)  # Track image filenames


features = torch.cat(features).numpy()

scaler = StandardScaler()
features = scaler.fit_transform(features)

pca = PCA(n_components=2)
features = pca.fit_transform(features)

n_clusters = 2
#kmeans = KMeans(n_clusters=n_clusters, init="random", max_iter=300, n_init=100)
#clusters = kmeans.fit_predict(features)
gmm = GaussianMixture(
    n_components=n_clusters,
    covariance_type="full",
    n_init=100,
    init_params="kmeans",
    max_iter=100,
)
clusters = gmm.fit_predict(features)

# Plotting the 2D PCA results with clusters
manual_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
if n_clusters > len(manual_colors):
    raise ValueError("Not enough colors in the manual color list for the number of clusters.")

plt.figure(figsize=(10, 7))
for i in range(n_clusters):
    cluster_points = features[clusters == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=manual_colors[i], label=f'Cluster {i}', alpha=0.6)
plt.title('2D PCA of Encoded Features')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.savefig('byol_features_pca.png')  # Save the PCA plot

# Output clustering results to folders
output_dir = 'byol_output_clusters'
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

# Create subdirectories for each cluster and save images to the corresponding cluster
for i in range(n_clusters):
    os.makedirs(os.path.join(output_dir, f'cluster_{i}'), exist_ok=True)

# Move images into their corresponding cluster folder
for idx, cluster_id in enumerate(clusters):
    img_path = os.path.join("../2", image_filenames[idx])
    img = Image.open(img_path)
    img.save(os.path.join(output_dir, f'cluster_{cluster_id}', image_filenames[idx]))

print("Clustered images saved to respective folders.")
