import torch
import torch.nn as nn
import os
from PIL import Image
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
import shutil

image_folder = '../2'
output_folder = 'autoenc_output_clusters'
model_save_path = 'autoencoder_model.pth'
comparison_folder = 'comparison_images'

def clear_and_create_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)  
    os.makedirs(folder_path) 

clear_and_create_folder(output_folder)
clear_and_create_folder(comparison_folder)

resize_transform = transforms.Compose([
    transforms.Resize((8,8)),  
    transforms.ToTensor(),
])

def load_images_from_folder(folder):
    images = []
    filenames = []
    original_images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path).convert('RGB')
        original_images.append(img)  
        img_transformed = resize_transform(img)  
        images.append(img_transformed)
        filenames.append(filename)  
    return torch.stack(images), filenames, original_images

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder_conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.encoder_relu1 = nn.ReLU()
        self.encoder_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_conv2 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1)
        self.encoder_relu2 = nn.ReLU()
        self.encoder_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.decoder_conv_trans1 = nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder_relu1 = nn.ReLU()
        self.decoder_conv_trans2 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder_sigmoid = nn.Sigmoid()
         
    def forward(self, x):
        x = self.encoder_conv1(x)
        x = self.encoder_relu1(x)
        x = self.encoder_pool1(x)
        x = self.encoder_conv2(x)
        x = self.encoder_relu2(x)
        x = self.encoder_pool2(x)
        x = self.decoder_conv_trans1(x)
        x = self.decoder_relu1(x)
        x = self.decoder_conv_trans2(x)
        x = self.decoder_sigmoid(x)
        return x

def load_model():
    autoencoder = ConvAutoencoder()
    autoencoder.load_state_dict(torch.load(model_save_path, weights_only=True))
    autoencoder.eval()
    return autoencoder

def extract_encoded_features(autoencoder):
    X_tensor, filenames, original_images = load_images_from_folder(image_folder)
    X_tensor = X_tensor.float()

    with torch.no_grad():
        x = autoencoder.encoder_conv1(X_tensor)
        x = autoencoder.encoder_relu1(x)
        x = autoencoder.encoder_pool1(x)
        x = autoencoder.encoder_conv2(x)
        x = autoencoder.encoder_relu2(x)
        x = autoencoder.encoder_pool2(x)
        encoded_features = x.view(len(X_tensor), -1).numpy()

    print(f'Encoded Features Shape: {encoded_features.shape}')
    return encoded_features, filenames, original_images

def perform_clustering(encoded_features):
    pca = PCA(n_components=2)
    encoded_features_2d = pca.fit_transform(encoded_features)

    gmm = GaussianMixture(
        n_components=2,
        covariance_type="full",
        n_init=100,
        init_params="kmeans",
        max_iter=100,
    )
    clusters = gmm.fit_predict(encoded_features_2d)

    n_clusters = len(set(clusters))
    manual_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    if n_clusters > len(manual_colors):
        raise ValueError("Not enough colors in the manual color list for the number of clusters.")
    plt.figure(figsize=(10, 7))
    for i in range(n_clusters):
        cluster_points = encoded_features_2d[clusters == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=manual_colors[i], label=f'Cluster {i}', alpha=0.6)

    plt.title('2D PCA of Encoded Features')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.savefig('encoded_features_pca.png')  

    return clusters

def save_images_with_clusters(clusters, original_images, filenames):
    def save_image_with_cluster(original_image, filename, cluster_label):
        cluster_folder = os.path.join(output_folder, f'cluster_{cluster_label}')
        os.makedirs(cluster_folder, exist_ok=True)
        output_path = os.path.join(cluster_folder, filename)
        original_image.save(output_path)

    for i, (original_image, filename, cluster_label) in enumerate(zip(original_images, filenames, clusters)):
        save_image_with_cluster(original_image, filename, cluster_label)

def reconstruct_and_compare(autoencoder):
    X_tensor, filenames, _ = load_images_from_folder(image_folder)
    X_tensor = X_tensor.float()

    def reconstruct_images(images, autoencoder):
        with torch.no_grad():
            reconstructed = autoencoder(images)
        return reconstructed

    def save_image_comparison(original_image, reconstructed_image, filename):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        
        orig_img = ToPILImage()(original_image)
        recon_img = ToPILImage()(reconstructed_image)
        
        ax[0].imshow(orig_img)
        ax[0].set_title('Original')
        ax[0].axis('off')
        
        ax[1].imshow(recon_img)
        ax[1].set_title('Reconstructed')
        ax[1].axis('off')
        
        plt.savefig(filename)
        plt.close()

    num_images = 5
    images_to_compare = []

    for cluster in range(2): 
        cluster_indices = [i for i, label in enumerate(clusters) if label == cluster]
        selected_indices = cluster_indices[:num_images]  
        images_to_compare.extend(selected_indices)

    images_to_compare_tensor = X_tensor[images_to_compare]
    reconstructed_images = reconstruct_images(images_to_compare_tensor, autoencoder)

    for i, idx in enumerate(images_to_compare):
        original_image = X_tensor[idx]
        reconstructed_image = reconstructed_images[i]
        save_image_comparison(original_image, reconstructed_image, os.path.join(comparison_folder, f'comparison_{i}.png'))

    print(f'Saved comparison images for {len(images_to_compare)} examples.')

if __name__ == "__main__":
    autoencoder = load_model()
    encoded_features, filenames, original_images = extract_encoded_features(autoencoder)
    clusters = perform_clustering(encoded_features)
    save_images_with_clusters(clusters, original_images, filenames)
    reconstruct_and_compare(autoencoder)
