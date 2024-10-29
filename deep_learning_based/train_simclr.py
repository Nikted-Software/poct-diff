import torch
from lightly import loss
from transform import SimCLRTransform
from lightly.data import LightlyDataset
from lightly.models.modules import heads
import matplotlib.pyplot as plt
import torch.optim as optim
from torch import nn



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
transform = SimCLRTransform(input_size=16,
        cj_prob  = 0,
        cj_strength=0.3,
        cj_bright = 0.1,
        cj_contrast = 0.1,
        cj_sat = 0.1,
        cj_hue = 0.1,
        random_gray_scale = 0,
        gaussian_blur = 0,
        vf_prob = 0.5,
        hf_prob = 0.5,
        rr_prob = 0.5,
        rr_degrees = (-180.0, 180.0),
        )

dataset = LightlyDataset(input_dir="images", transform=transform)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
)

criterion = loss.NTXentLoss(temperature=0.5)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

best_loss = float('inf')  

for epoch in range(30):
    total_loss = 0.0  
    for (view0, view1), targets, filenames in dataloader:
        optimizer.zero_grad()
        

        #print(f"Filenames: {filenames}")
        #print(f"View 0 shape: {view0.shape}, View 1 shape: {view1.shape}")


        #view0_images = view0.permute(0, 2, 3, 1).numpy() 
        #view1_images = view1.permute(0, 2, 3, 1).numpy()
        #plt.figure(figsize=(12, 6))
        #plt.subplot(2, 1, 1)
        #plt.imshow(view0_images[0])
        #plt.axis('off')
        #plt.title('View 0')
        #plt.subplot(2, 1, 2)
        #plt.imshow(view1_images[0])
        #plt.axis('off')
        #plt.title('View 1')
        #plt.show()
        
        z0 = model(view0)
        z1 = model(view1)
        loss_value = criterion(z0, z1)
        loss_value.backward()
        optimizer.step()
        total_loss += loss_value.item()  

    avg_loss = total_loss / len(dataloader)
    scheduler.step(avg_loss)
    print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), 'simclr_best_model.pth')
        print('Best model saved to simclr_best_model.pth')