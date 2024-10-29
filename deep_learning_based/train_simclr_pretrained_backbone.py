import torch
import torchvision
from lightly import loss
from transform import SimCLRTransform
from lightly.data import LightlyDataset
from lightly.models.modules import heads
import matplotlib.pyplot as plt
import torch.optim as optim

class SimCLR(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = heads.SimCLRProjectionHead(
            input_dim=512, 
            hidden_dim=512,
            output_dim=128,
        )

    def forward(self, x):
        features = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(features)
        return z

    def forward(self, x):
        features = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(features)
        return z


backbone = torchvision.models.resnet18()
backbone.fc = torch.nn.Identity() 

#for param in backbone.parameters():
#    param.requires_grad = False

model = SimCLR(backbone)
transform = SimCLRTransform(input_size=16,
        cj_prob  = 0.5,
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
    batch_size=32,
    shuffle=True,
)

criterion = loss.NTXentLoss(temperature=0.5)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

best_loss = float('inf')  

for epoch in range(20):
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