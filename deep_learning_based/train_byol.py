import torch
from torch import nn
import torch.optim as optim
from lightly.data import LightlyDataset
from transform import SimCLRTransform
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
import copy
from lightly.loss import NegativeCosineSimilarity
import random
import numpy as np
import matplotlib.pyplot as plt
from lightly.utils.scheduler import cosine_schedule

input_size = 16

random_seed = 7
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
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
momentum_encoder = BYOL(Backbone())

momentum_encoder.load_state_dict(model.state_dict())

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
momentum_encoder.to(device)

transform = SimCLRTransform(input_size=input_size,
                            cj_prob=0,
                            random_gray_scale=0,
                            gaussian_blur=0.5,
                            vf_prob=0.5,
                            hf_prob=0.5,
                            rr_prob=0.7,
                            rr_degrees=(-180.0, 180.0))

dataset = LightlyDataset("images", transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

criterion = NegativeCosineSimilarity()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)

epochs = 20
best_loss = float('inf')
best_model_path = "best_byol_model.pth"

print("Starting Training")
for epoch in range(epochs):
    total_loss = 0
    
    momentum_val = cosine_schedule(epoch, epochs, 0.996, 1)
    
    model.train()
    for batch in dataloader:
        x0, x1 = batch[0]
        
        # Update the momentum encoder (EMA) using the cosine schedule
        update_momentum(model.backbone, model.backbone_momentum, m=momentum_val)
        update_momentum(model.projection_head, model.projection_head_momentum, m=momentum_val)

        x0 = x0.to(device)
        x1 = x1.to(device)


        # Student predictions
        p0 = model(x0)
        p1 = model(x1)

        # Teacher projections (momentum encoder)
        z0 = model.forward_momentum(x0)
        z1 = model.forward_momentum(x1)

        loss = 0.5 * (criterion(p0, z1) + criterion(p1, z0))
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
