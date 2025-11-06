# Ứng dụng Nhận dạng Thức ăn AI

Ứng dụng nhận dạng thức ăn sử dụng trí tuệ nhân tạo (AI) để phân loại và cung cấp thông tin về các món ăn Việt Nam.

## 🌟 Tính năng chính

- **Nhận dạng thức ăn**: Sử dụng mô hình CNN để nhận dạng 17 loại thức ăn Việt Nam
- **Thông tin chi tiết**: Cung cấp giá, calo, loại món ăn và điểm sức khỏe
- **Nhận dạng khay ăn**: Phân tích 5 ngăn khay ăn và nhận dạng từng món
- **API RESTful**: Giao diện API đầy đủ cho tích hợp với ứng dụng khác
- **Giao diện web**: Giao diện người dùng thân thiện

## 🍽️ Các món ăn được hỗ trợ

1. Canh chua có cá
2. Canh chua không cá
3. Canh rau cải
4. Canh rau muống
5. Cá hú kho
6. Cơm trắng
7. Rau củ sắn xào
8. Không có món ăn nào cả
9. Lagim sàu
10. Sườn nướng
11. Thịt kho
12. Thịt Kho Trứng
13. Thịt Kho 2 Trứng
14. Trứng chiên
15. Đậu hũ sốt cà
16. Rau đậu que xào
17. Rau đậu đũa xào

## 🚀 Cài đặt và chạy

### Yêu cầu hệ thống

- Python 3.7+
- TensorFlow 2.x
- Flask
- OpenCV
- PIL (Pillow)
- NumPy

### Cài đặt

1. **Clone repository**:
```bash
git clone <repository-url>
cd food-recognition-app
```

2. **Cài đặt dependencies**:
```bash
pip install -r requirements.txt
```

3. **Đảm bảo có các file cần thiết**:
   - `final_model.h5` - Mô hình AI đã huấn luyện
   - `cnn_classes.txt` - Danh sách các lớp thức ăn
   - `person_info.json` - Thông tin chi tiết về thức ăn

### Chạy ứng dụng

```bash
python app.py
```

Ứng dụng sẽ chạy tại: `http://localhost:5000`

## 📖 Cách sử dụng

### 1. Giao diện web

Truy cập `http://localhost:5000` để sử dụng giao diện web:
- Tải lên hình ảnh thức ăn
- Xem kết quả nhận dạng
- Thông tin chi tiết về món ăn

### 2. API Endpoints

#### Nhận dạng thức ăn đơn lẻ
```bash
POST /api/predict
Content-Type: application/json

{
    "image": "data:image/jpeg;base64,..."
}
```

#### Nhận dạng khay ăn (5 ngăn)
```bash
POST /api/predict-tray
Content-Type: application/json

{
    "image": "data:image/jpeg;base64,...",
    "padding_ratio": 0.02,
    "min_confidence": 0.6
}
```

#### Cắt hình ảnh thành 5 phần
```bash
POST /api/crop-image
Content-Type: application/json

{
    "image": "data:image/jpeg;base64,..."
}
```

#### Kiểm tra sức khỏe hệ thống
```bash
GET /api/health
```

#### Lấy danh sách lớp thức ăn
```bash
GET /api/classes
```

#### Lấy thông tin chi tiết thức ăn
```bash
GET /api/food-info
```

## 🧪 Kiểm tra

Chạy script kiểm tra để đảm bảo ứng dụng hoạt động bình thường:

```bash
python test_app.py
```

## 📊 Cấu trúc dự án

```
food-recognition-app/
├── app.py                 # Ứng dụng Flask chính
├── test_app.py           # Script kiểm tra
├── final_model.h5        # Mô hình AI
├── cnn_classes.txt       # Danh sách lớp
├── person_info.json      # Thông tin thức ăn
├── nhan_dien.html        # Giao diện nhận dạng
├── menu.html            # Giao diện menu
├── templates/           # Template HTML
├── uploads/            # Thư mục tải lên
├── app.log            # File log
└── README_VI.md       # Hướng dẫn này
```

## 🔧 Cấu hình

### Biến môi trường

- `UPLOAD_FOLDER`: Thư mục lưu file tải lên (mặc định: 'uploads')
- `MODEL_PATH`: Đường dẫn đến mô hình (mặc định: 'final_model.h5')
- `CLASSES_PATH`: Đường dẫn đến file lớp (mặc định: 'cnn_classes.txt')

### Tối ưu hiệu suất

- **Cache dự đoán**: Tự động cache kết quả dự đoán để tăng tốc
- **Quản lý bộ nhớ**: Tự động dọn dẹp bộ nhớ định kỳ
- **GPU support**: Tự động phát hiện và sử dụng GPU nếu có

## 📝 Log và Debug

### Xem log

```bash
tail -f app.log
```

### Mức độ log

- `INFO`: Thông tin chung
- `WARNING`: Cảnh báo
- `ERROR`: Lỗi
- `DEBUG`: Thông tin debug chi tiết

## 🛠️ Phát triển

### Thêm món ăn mới

1. Cập nhật `cnn_classes.txt`
2. Thêm thông tin vào `person_info.json`
3. Huấn luyện lại mô hình

### Tùy chỉnh mô hình

Chỉnh sửa các tham số trong `app.py`:
- `_cache_max_size`: Kích thước cache
- `min_confidence`: Ngưỡng tin cậy tối thiểu
- `padding_ratio`: Tỷ lệ padding cho khay ăn

## 🐛 Xử lý sự cố

### Lỗi thường gặp

1. **Model không tải được**:
   - Kiểm tra file `final_model.h5` có tồn tại
   - Kiểm tra quyền truy cập file

2. **Lỗi nhận dạng**:
   - Kiểm tra chất lượng hình ảnh
   - Điều chỉnh `min_confidence`

3. **Lỗi bộ nhớ**:
   - Giảm `_cache_max_size`
   - Tăng tần suất dọn dẹp bộ nhớ

### Debug

Bật chế độ debug trong `app.py`:
```python
app.run(debug=True)
```

## 📞 Hỗ trợ

Nếu gặp vấn đề, vui lòng:
1. Kiểm tra file log `app.log`
2. Chạy script kiểm tra `test_app.py`
3. Kiểm tra các file phụ thuộc

## 📄 Giấy phép

Dự án này được phát hành dưới giấy phép MIT.

## 🤝 Đóng góp

Mọi đóng góp đều được chào đón! Vui lòng:
1. Fork repository
2. Tạo branch mới
3. Commit thay đổi
4. Tạo Pull Request

---

**Lưu ý**: Ứng dụng này được thiết kế đặc biệt cho nhận dạng thức ăn Việt Nam và có thể cần điều chỉnh cho các loại thức ăn khác.
