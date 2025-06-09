import cv2
import os
from ultralytics import YOLO

# Load mô hình đã huấn luyện
model_path = os.path.join("runs", "detect", "yolo_inventory", "weights", "best.pt")
model = YOLO(model_path)

# Khởi tạo webcam
cap = cv2.VideoCapture(0)

print("Nhấn SPACE để chụp ảnh và detect")
print("Nhấn 'q' để thoát")

# Vòng lặp chính để đọc webcam
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Hiển thị webcam
    cv2.putText(frame, "Nhan SPACE de chup anh", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Webcam', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # Space để chụp
        # Detect
        results = model(frame, conf=0.5)
        
        # Đếm số lỗi và lưu thông tin
        detected_count = 0
        detected_labels = []
        if results[0].boxes is not None:
            detected_count = len(results[0].boxes)
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0])
                detected_labels.append(f"{label}({conf:.2f})")
        
        # Hiển thị kết quả
        if detected_count > 0:
            # Có lỗi - hiển thị ảnh có bounding box
            result_frame = results[0].plot()  # Sử dụng plot() của ultralytics để vẽ box tự động
            cv2.putText(result_frame, f"PHAT HIEN {detected_count} LOI!", 
                       (10, result_frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.imshow('Ket qua - Nhan phim bat ky de dong', result_frame)
            print(f"Phát hiện {detected_count} lỗi: {', '.join(detected_labels)}")
        else:
            # Không có lỗi
            cv2.putText(frame, "KHONG CO LOI!", 
                       (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            cv2.imshow('Ket qua - Nhan phim bat ky de dong', frame)
            print("Không phát hiện lỗi")
        
        cv2.waitKey(0)  # Chờ nhấn phím để đóng kết quả
        cv2.destroyWindow('Ket qua - Nhan phim bat ky de dong')
        
    elif key == ord('q'):
        break

# Giải phóng webcam và đóng tất cả cửa sổ
cap.release()
cv2.destroyAllWindows()
