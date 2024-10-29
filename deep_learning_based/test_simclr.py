import os
import torch
import torchvision
from lightly.data import LightlyDataset
from lightly.models.modules import heads
from sklearn.mixture import GaussianMixture
from PIL import Image
import shutil
from sklearn.decomposition import PCA
from transform import SimCLRTransform
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from torch import nn


random_number = 42

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
    
class SimCLR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.projection_head = heads.SimCLRProjectionHead(
            input_dim=128, 
            hidden_dim=64,
            output_dim=32,
        )

    def forward(self, x):
        features = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(features)
        return z



model = SimCLR()

model.load_state_dict(torch.load('simclr_best_model.pth', weights_only=True))
model.eval()

transform = SimCLRTransform(input_size=16,
                            cj_prob=0,
                            random_gray_scale=0,
                            gaussian_blur=0,
                            vf_prob=0,
                            hf_prob=0,
                            rr_prob=0)

dataset = LightlyDataset(input_dir="2", transform=transform)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,)

features = []
image_filenames = []  

with torch.no_grad():
    for (view0, view1), targets, batch_filenames in dataloader:
        feature_output = model(view0)  
        features.append(feature_output.flatten(start_dim=1))
        image_filenames.extend(batch_filenames)  


features = torch.cat(features).numpy()
scaler = StandardScaler()
features = scaler.fit_transform(features)
pca = PCA(n_components=2)
features = pca.fit_transform(features)

n_clusters = 2
#kmeans = KMeans(n_clusters=n_clusters, init="random", max_iter=300, n_init=100)
#clusters = kmeans.fit_predict(features)

gmm = GaussianMixture(
        n_components=2,
        covariance_type="full",
        n_init=100,
        init_params="kmeans",
        max_iter=100,
    )
clusters = gmm.fit_predict(features)

n_clusters = len(set(clusters))
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
plt.savefig('simclr_features_pca.png')  

output_dir = 'simclr_output_clusters'
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)
for i in range(n_clusters):
    os.makedirs(os.path.join(output_dir, f'cluster_{i}'), exist_ok=True)
for idx, cluster_id in enumerate(clusters):
    img_path = os.path.join("2", image_filenames[idx]) 
    img = Image.open(img_path)
    img.save(os.path.join(output_dir, f'cluster_{cluster_id}', image_filenames[idx]))
print("Clustered images saved to respective folders.")
