import os
from ultralytics import YOLO

# 1. Đường dẫn dataset (local path for VS Code)  
dataset_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "dataset_yolo")
data_yaml_path = os.path.join(dataset_dir, "data.yaml")

print(f"Dataset directory: {dataset_dir}")
print(f"Data config file: {data_yaml_path}")

# 2. Load YOLO model và train
model = YOLO('yolov8n.pt')  # Load pretrained YOLOv8 nano model

# 3. Train the model
results = model.train(
    data=data_yaml_path,
    epochs=50,
    imgsz=640,
    batch=8,
    name='yolo_inventory'
)

print("Training completed!")