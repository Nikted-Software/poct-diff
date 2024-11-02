import os
from tqdm import *
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn import Parameter
from torchvision import transforms
from torchvision.utils import save_image
from sklearn.cluster import KMeans
import numpy as np
from tqdm import *
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from PIL import Image
import torch.nn.functional as F
import shutil

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder_conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.encoder_relu1 = nn.ReLU()
        self.encoder_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_conv2 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1)
        self.encoder_relu2 = nn.ReLU()
        self.encoder_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.decoder_conv_trans1 = nn.ConvTranspose2d(
            8, 16, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.decoder_relu1 = nn.ReLU()
        self.decoder_conv_trans2 = nn.ConvTranspose2d(
            16, 3, kernel_size=3, stride=2, padding=1, output_padding=1
        )
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


class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters=2, hidden=10, cluster_centers=None, alpha=1.0):
        super(ClusteringLayer, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.hidden = hidden
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.n_clusters, self.hidden, dtype=torch.float
            ).to(device)
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, x):
        norm_squared = torch.sum((x.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator**power
        t_dist = (numerator.t() / torch.sum(numerator, 1)).t()
        return t_dist


class DEC(nn.Module):
    def __init__(
        self,
        n_clusters=2,
        autoencoder=None,
        hidden=10,
        cluster_centers=None,
        alpha=1.0,
    ):
        super(DEC, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.hidden = hidden
        self.cluster_centers = cluster_centers
        self.autoencoder = autoencoder
        self.clusteringlayer = ClusteringLayer(
            self.n_clusters, self.hidden, self.cluster_centers, self.alpha
        )

    def target_distribution(self, q_):
        weight = (q_**2) / torch.sum(q_, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    def forward(self, x):
        x = self.autoencoder(x)
        x = x.view(-1, 3 * 8 * 8)
        return self.clusteringlayer(x)

    def visualize(self, epoch, x):
        fig = plt.figure()
        ax = plt.subplot(111)
        x = self.autoencoder(x).detach()
        x = x.view(-1, 3 * 8 * 8)
        x = x.cpu().numpy()[:2000]
        x_embedded = TSNE(n_components=2).fit_transform(x)
        plt.scatter(x_embedded[:, 0], x_embedded[:, 1])
        fig.savefig("plots/mnist_{}.png".format(epoch))
        plt.close(fig)


def save_checkpoint(state, filename, is_best):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print("=> Saving new checkpoint")
        torch.save(state, filename)
    else:
        print("=> Validation Accuracy did not improve")


def pretrain(**kwargs):
    data = kwargs["data"]
    model = kwargs["model"]
    num_epochs = kwargs["num_epochs"]
    savepath = kwargs["savepath"]
    checkpoint = kwargs["checkpoint"]
    start_epoch = checkpoint["epoch"]
    parameters = list(autoencoder.parameters())
    optimizer = torch.optim.Adam(parameters, lr=1e-4)
    train_loader = DataLoader(dataset=data, batch_size=16, shuffle=True)
    for epoch in range(start_epoch, num_epochs):
        for data in train_loader:

            img = data.float()
            img = img.to(device)
            output = model(img)
            loss = nn.MSELoss()(output, img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(
            "epoch [{}/{}], MSE_loss:{:.4f}".format(epoch + 1, num_epochs, loss.item())
        )
        state = loss.item()
        is_best = False
        if state < checkpoint["best"]:
            checkpoint["best"] = state
            is_best = True

        save_checkpoint(
            {"state_dict": model.state_dict(), "best": state, "epoch": epoch},
            savepath,
            is_best,
        )


def train(**kwargs):
    data = kwargs["data"]
    model = kwargs["model"]
    num_epochs = kwargs["num_epochs"]
    savepath = kwargs["savepath"]
    checkpoint = kwargs["checkpoint"]
    start_epoch = checkpoint["epoch"]
    features = []
    train_loader = DataLoader(dataset=data, batch_size=16, shuffle=True)

    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32)

    for i, batch in enumerate(train_loader):

        if isinstance(batch, np.ndarray):
            img = torch.tensor(batch, dtype=torch.float32).to(device)
        else:
            img = batch.float().view(-1, 3, 8, 8).to(device)

        imgs_out = autoencoder(img).detach().cpu()

        for img_out in imgs_out:
            features.append(img_out.unsqueeze(0))

    features = torch.cat(features, dim=0).view(len(features), -1)
    # ============K-means=======================================
    kmeans = KMeans(n_clusters=2, random_state=0).fit(features)
    cluster_centers = kmeans.cluster_centers_
    cluster_centers = torch.tensor(cluster_centers, dtype=torch.float).to(device)
    model.clusteringlayer.cluster_centers = torch.nn.Parameter(cluster_centers)
    # =========================================================

    loss_function = nn.KLDivLoss(size_average=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)

    print("Training")
    for epoch in range(start_epoch, num_epochs):
        batch = data
        img = batch.view(-1, 3, 8, 8).float()
        img = img.to(device)

        output = F.log_softmax(model(img), dim=1)
        target = model.target_distribution(output).detach()

        epsilon = 1e-10
        target = target + epsilon
        target = target / target.sum(dim=1, keepdim=True)

        loss = loss_function(output, target)
        if epoch % 50 == 0:
            print("plotting")
            dec.visualize(epoch, img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = loss.item()
        print(state)
        is_best = False
        if abs(state) < abs(checkpoint["best"]):
            checkpoint["best"] = abs(state)
            is_best = True

        save_checkpoint(
            {"state_dict": model.state_dict(), "best": state, "epoch": epoch},
            savepath,
            is_best,
        )


def load_images_from_folder(folder_path):
    transform = transforms.Compose(
        [
            transforms.Resize((8, 8)),
            transforms.ToTensor(),
        ]
    )

    image_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(folder_path, filename)
            image = Image.open(img_path).convert("RGB")
            image = transform(image)
            image_list.append(image)

    x = torch.stack(image_list)

    return x


if __name__ == "__main__":
    
    output_dir = "output"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    output_dir = "saves"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs("saves")
        
    import argparse

    parser = argparse.ArgumentParser(
        description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--pretrain_epochs", default=100, type=int)
    parser.add_argument("--train_epochs", default=300, type=int)
    parser.add_argument("--save_dir", default="saves")
    args = parser.parse_args()
    print(args)
    epochs_pre = args.pretrain_epochs
    batch_size = args.batch_size

    x = load_images_from_folder("images")
    autoencoder = ConvAutoencoder().to(device)
    ae_save_path = "saves/sim_autoencoder.pth"

    if os.path.isfile(ae_save_path):
        print("Loading {}".format(ae_save_path))
        checkpoint = torch.load(ae_save_path)
        autoencoder.load_state_dict(checkpoint["state_dict"])
    else:
        print("=> no checkpoint found at '{}'".format(ae_save_path))
        checkpoint = {"epoch": 0, "best": float("inf")}
    pretrain(
        data=x,
        model=autoencoder,
        num_epochs=epochs_pre,
        savepath=ae_save_path,
        checkpoint=checkpoint,
    )

    dec_save_path = "saves/dec.pth"
    dec = DEC(
        n_clusters=2,
        autoencoder=autoencoder,
        hidden=2,
        cluster_centers=None,
        alpha=1.0,
    ).to(device)
    if os.path.isfile(dec_save_path):
        print("Loading {}".format(dec_save_path))
        checkpoint = torch.load(dec_save_path)
        dec.load_state_dict(checkpoint["state_dict"])
    else:
        print("=> no checkpoint found at '{}'".format(dec_save_path))
        checkpoint = {"epoch": 0, "best": float("inf")}
    train(
        data=x,
        model=dec,
        num_epochs=args.train_epochs,
        savepath=dec_save_path,
        checkpoint=checkpoint,
    )

    

    class ImageFolderDataset(Dataset):
        def __init__(self, folder_path, transform=None):
            self.folder_path = folder_path
            self.transform = transform
            self.image_paths = [
                os.path.join(folder_path, fname)
                for fname in os.listdir(folder_path)
                if fname.endswith((".png", ".jpg", ".jpeg"))
            ]

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, image_path

    transform = transforms.Compose([
        transforms.Resize((8, 8)),
        transforms.ToTensor(),
    ])

    folder_path = "2"
    dataset = ImageFolderDataset(folder_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dec.to(device)

    all_assignments = []
    with torch.no_grad():
        for img, path in dataloader:
            img = img.to(device)
            q = dec(img)
            _, cluster_assignment = torch.max(q, dim=1)
            all_assignments.append((path[0], cluster_assignment.item()))

    embeddings = []
    with torch.no_grad():
        for img, _ in dataloader:
            img = img.to(device)
            encoded = dec.autoencoder(img).cpu()
            embeddings.append(encoded.view(encoded.size(0), -1))

    embeddings = torch.cat(embeddings, dim=0).numpy()
    kmeans = KMeans(n_clusters=2, random_state=0).fit(embeddings)

    def save_images_to_cluster_folders(assignments, base_folder_name):
        for path, cluster in assignments:
            cluster_folder = os.path.join(base_folder_name, str(cluster))
            os.makedirs(cluster_folder, exist_ok=True)

            image_name = os.path.basename(path)
            destination = os.path.join(cluster_folder, image_name)
            shutil.copy2(path, destination)

    dec_assignments = [(path, dec_cluster) for path, dec_cluster in all_assignments]

    kmeans_assignments = [
        (all_assignments[i][0], kmeans_cluster) for i, kmeans_cluster in enumerate(kmeans.labels_)
    ]

    save_images_to_cluster_folders(dec_assignments, "output/DEC")
    save_images_to_cluster_folders(kmeans_assignments, "output/KMeans")

    print("Images saved to respective DEC and KMeans cluster folders.")

