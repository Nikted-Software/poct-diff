import torch
from torch import nn
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import SimSiamPredictionHead, SimSiamProjectionHead
from lightly.data import LightlyDataset
from transform import SimCLRTransform
import torch.optim as optim

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

class simSiam(nn.Module):
    def __init__(self):
        super(simSiam, self).__init__()
        self.backbone = Backbone()
        self.projection_head = SimSiamProjectionHead(8 * 4 * 4, 32, 8)  
        self.prediction_head = SimSiamPredictionHead(8, 4, 8)

    def forward(self, x):
        f = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(f)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p

model = simSiam()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

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

dataset = LightlyDataset("images", transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

criterion = NegativeCosineSimilarity()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)
epochs = 30

best_loss = float('inf')
best_model_path = "best_simsiam_model.pth"

print("Starting Training")
for epoch in range(epochs):
    total_loss = 0
    model.train()  

    for batch in dataloader:
        x0, x1 = batch[0]
        x0 = x0.to(device)
        x1 = x1.to(device)

        z0, p0 = model(x0)
        z1, p1 = model(x1)

        loss = 0.5 * (criterion(z0, p1) + criterion(z1, p0))
        total_loss += loss.detach().item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch: {epoch + 1:>02}, Loss: {avg_loss:.5f}")
    scheduler.step(avg_loss)

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved with loss: {best_loss:.5f}")

print("Training complete.")
