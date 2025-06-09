# 🔍 Hệ Thống Phát Hiện Lỗi - Defect Detection System

## 📋 Mô tả
Hệ thống sử dụng AI (YOLOv8) để phát hiện tự động các lỗi trên sản phẩm với giao diện web thân thiện.

## 🎯 Tính năng chính
- ✅ **Phát hiện 4 loại lỗi:**
  - **Be** (Bể)
  - **Mop-lom** (Móp lõm)
  - **Nut** (Nứt) 
  - **Tray_xuoc** (Trầy xước)

- ✅ **Tính năng giao diện:**
  - Camera trực tiếp
  - Upload ảnh từ máy tính
  - Hiển thị bounding boxes với nhãn
  - Lịch sử phát hiện chi tiết
  - Độ tin cậy cho mỗi phát hiện

## 📁 Cấu trúc file chính
```
web3/
├── app.py               # Server Flask với YOLO model (CHÍNH)
├── frontend.html        # Giao diện web hoạt động ổn định (CHÍNH)
├── Start_System.bat     # File khởi động hệ thống
└── README.md           # Hướng dẫn này
```

## 🚀 Cách chạy hệ thống

### Phương pháp 1: File batch (Đơn giản nhất)
1. Double-click vào file `Start_System.bat`
2. Đợi server khởi động (khoảng 10-15 giây)
3. Mở trình duyệt tại: http://localhost:5000

### Phương pháp 2: Command line
```powershell
cd c:\project_root\web3
D:\python.exe app.py
```

## 🖥️ Sử dụng giao diện

### 📸 Sử dụng Camera
1. Click **"Bật Camera"** để khởi động camera
2. Click **"Chụp Ảnh"** để chụp ảnh từ camera
3. Click **"Kiểm Tra Lỗi"** để phân tích

### 📁 Upload ảnh
1. Click **"Choose File"** để chọn ảnh từ máy tính
2. Click **"Tải Ảnh Lên"** 
3. Click **"Kiểm Tra Lỗi"** để phân tích

### 📊 Kết quả hiển thị
- **Ảnh với bounding boxes**: Khung đỏ quanh vùng lỗi
- **Tên lỗi cụ thể**: Thay vì "Unknown"
- **Độ tin cậy**: Phần trăm chính xác
- **Lịch sử**: Tất cả các lần phát hiện trước đó

## 🧪 Test với ảnh mẫu
Sử dụng ảnh test từ thư mục:
```
c:\project_root\dataset\dataset_yolo\images\val\
```

## 🛠️ Yêu cầu hệ thống
- Python 3.8+
- YOLO model trained: `c:\project_root\runs\detect\yolo_inventory\weights\best.pt`
- Các thư viện: Flask, OpenCV, Ultralytics, NumPy

## 🔧 Xử lý sự cố

### Server không khởi động
- Kiểm tra Python path: `D:\python.exe`
- Kiểm tra model file tồn tại tại đường dẫn đã chỉ định

### Camera không hoạt động
- Cấp quyền camera cho trình duyệt
- Thử sử dụng upload ảnh thay thế

### Lỗi connection
- Kiểm tra server đang chạy tại port 5000
- Tắt firewall hoặc antivirus tạm thời

## 📈 Kết quả đã test
- ✅ Phát hiện thành công lỗi "Tray_xuoc" (4 lỗi)
- ✅ Hiển thị bounding boxes chính xác
- ✅ Lưu lịch sử với chi tiết đầy đủ

## 🔄 Dừng hệ thống
- Đóng cửa sổ terminal hoặc
- Nhấn `Ctrl+C` trong terminal


