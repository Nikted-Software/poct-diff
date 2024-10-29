import torch
import torch.nn as nn
import torch.optim as optim
import os
from PIL import Image
import torchvision.transforms as transforms


image_folder = 'images'
model_save_path = 'autoencoder_model.pth'


resize_transform = transforms.Compose([
    transforms.Resize((8,8)),  
    transforms.ToTensor(),
])


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path).convert('RGB')
        img_transformed = resize_transform(img)  
        images.append(img_transformed)
    return torch.stack(images)

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

# Training the autoencoder
def train_autoencoder():
    X_tensor = load_images_from_folder(image_folder)
    print(f'Image Tensor Shape: {X_tensor.shape}')
    X_tensor = X_tensor.float()

    autoencoder = ConvAutoencoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)

    epochs = 100
    batch_size = 16

    for epoch in range(epochs):
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size]

            # Forward pass
            output = autoencoder(batch)
            loss = criterion(output, batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    torch.save(autoencoder.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

if __name__ == "__main__":
    train_autoencoder()
