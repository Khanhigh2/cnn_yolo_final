import torch
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import os

# -------- 1. Load lại model đã huấn luyện ----------
model_path = os.path.join(os.path.dirname(__file__), "model_cnn_inventory.pth")

# Danh sách class bạn đã dùng để huấn luyện
classes = ['lo_vi_song', 'may_giat', 'may_hut_bui', 'noi_com_dien', 'tivi', 'tu_lanh']

# Tạo model
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# -------- 2. Tiền xử lý ảnh ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

def predict_image(image_path):
    # Load ảnh và transform
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)  # thêm batch dimension

    # Dự đoán
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        class_name = classes[predicted.item()]
        confidence = torch.softmax(outputs, 1)[0][predicted.item()].item()

    return class_name, confidence

# -------- 3. Dùng thử 1 ảnh ----------
image_path = r"C:\project_root\dataset\dataset_classify\train\lo_vi_song\lo_vi_song_1.jpg"  # sửa thành ảnh của bạn
label, confidence = predict_image(image_path)
print(f"Nhận diện: {label} (độ tin cậy: {confidence*100:.2f}%)")