import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader

# 1. Đường dẫn dataset (local path for VS Code)
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "dataset_classify")
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")

# 2. Transform ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

# Kiểm tra đường dẫn dataset
if not os.path.exists(train_dir):
    print(f"Error: Training directory not found: {train_dir}")
    print("Please make sure the dataset is in the correct location.")
    exit(1)

if not os.path.exists(val_dir):
    print(f"Error: Validation directory not found: {val_dir}")
    print("Please make sure the dataset is in the correct location.")
    exit(1)

# In thông tin đường dẫn dataset
print(f"Dataset paths:")
print(f"  Training: {train_dir}")
print(f"  Validation: {val_dir}")

# 3. Load dataset
train_set = datasets.ImageFolder(train_dir, transform=transform)
val_set = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
val_loader = DataLoader(val_set, batch_size=8)

# In thông tin class để kiểm tra
print("Classes:", train_set.classes)
print("Number of classes:", len(train_set.classes))

# 4. Tạo model
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(train_set.classes))  # Đặt đúng số class

# 5. Setup training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. Training loop
for epoch in range(6):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1} loss: {running_loss / len(train_loader):.4f}")

# 7. Validation
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Validation accuracy: {100 * correct / total:.2f}%")

# 8. Save model (local path for VS Code)
model_save_path = os.path.join(os.path.dirname(__file__), "model_cnn_inventory.pth")
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to: {model_save_path}")
