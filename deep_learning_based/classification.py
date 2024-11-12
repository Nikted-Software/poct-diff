import shutil
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import copy


transform = transforms.Compose(
    [transforms.Resize((16, 16)), transforms.ToTensor()])

train_dataset = datasets.ImageFolder(
    root="classification_images/train", transform=transform
)
val_dataset = datasets.ImageFolder(
    root="classification_images/val", transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)


class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = SmallCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_loss = float("inf")
best_model = None

for epoch in range(10):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(
        f"Epoch {epoch+1}, Training Loss: {running_loss /
                                           len(train_loader)}, Validation Loss: {val_loss}"
    )

    if val_loss < best_loss:
        best_loss = val_loss
        best_model = copy.deepcopy(model.state_dict())
        torch.save(best_model, "best_classification_model.pth")


transform = transforms.Compose(
    [transforms.Resize((16, 16)), transforms.ToTensor()])


class UnlabeledImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = [
            f
            for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f))
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.image_files[idx]


model.load_state_dict(torch.load("best_classification_model.pth"))
model.eval()

output_folder = "classified_images"
class_0_folder = os.path.join(output_folder, "Class_0")
class_1_folder = os.path.join(output_folder, "Class_1")
os.makedirs(class_0_folder, exist_ok=True)
os.makedirs(class_1_folder, exist_ok=True)

test_dataset = UnlabeledImageDataset(folder_path="2", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

with torch.no_grad():
    for images, filenames in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        for filename, label in zip(filenames, predicted):
            source_path = os.path.join("2", filename)
            target_folder = class_0_folder if label.item() == 0 else class_1_folder
            shutil.copy(source_path, os.path.join(target_folder, filename))

print("Images saved in respective class folders.")
