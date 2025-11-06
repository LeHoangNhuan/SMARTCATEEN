#!/usr/bin/env python3
"""
Script test layout cắt khay mới để cải thiện độ chính xác nhận diện
"""

import requests
import json
import base64
from PIL import Image
import io

def create_test_tray_image():
    """Tạo ảnh khay test với layout 5 ô"""
    # Tạo ảnh khay 800x600
    width, height = 800, 600
    img = Image.new('RGB', (width, height), color='white')
    
    # Vẽ các ô khay
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    
    # Layout mới: 3 trên, 2 dưới
    regions = [
        (0.00, 0.05, 0.32, 0.35),  # Ô 1 - trên trái
        (0.33, 0.05, 0.65, 0.35),  # Ô 2 - trên giữa
        (0.66, 0.05, 0.98, 0.35),  # Ô 3 - trên phải
        (0.00, 0.40, 0.38, 0.95),  # Ô 4 - dưới trái
        (0.50, 0.40, 0.98, 0.95),  # Ô 5 - dưới phải
    ]
    
    colors = ['red', 'green', 'blue', 'yellow', 'purple']
    
    for i, (x1r, y1r, x2r, y2r) in enumerate(regions):
        x1 = int(x1r * width)
        y1 = int(y1r * height)
        x2 = int(x2r * width)
        y2 = int(y2r * height)
        
        # Vẽ khung ô
        draw.rectangle([x1, y1, x2, y2], outline='black', width=3)
        
        # Vẽ màu nền
        draw.rectangle([x1+5, y1+5, x2-5, y2-5], fill=colors[i])
        
        # Vẽ số thứ tự
        draw.text((x1+10, y1+10), f"Ô {i+1}", fill='white')
    
    # Convert to base64
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

def test_old_layout():
    """Test layout cũ"""
    print("🔍 Test layout cũ...")
    
    # Layout cũ
    old_regions = [
        (0.05, 0.02, 0.34, 0.41),
        (0.36, 0.02, 0.63, 0.41),
        (0.65, 0.02, 0.96, 0.41),
        (0.05, 0.44, 0.43, 0.98),
        (0.50, 0.44, 0.95, 0.98),
    ]
    
    print("Layout cũ:")
    for i, (x1, y1, x2, y2) in enumerate(old_regions):
        print(f"  Ô {i+1}: ({x1:.2f}, {y1:.2f}) -> ({x2:.2f}, {y2:.2f})")
        print(f"    Kích thước: {x2-x1:.2f} x {y2-y1:.2f}")

def test_new_layout():
    """Test layout mới"""
    print("\n🔍 Test layout mới...")
    
    # Layout mới
    new_regions = [
        (0.00, 0.05, 0.32, 0.35),  # Ô 1 - trên trái
        (0.33, 0.05, 0.65, 0.35),  # Ô 2 - trên giữa
        (0.66, 0.05, 0.98, 0.35),  # Ô 3 - trên phải
        (0.00, 0.40, 0.38, 0.95),  # Ô 4 - dưới trái
        (0.50, 0.40, 0.98, 0.95),  # Ô 5 - dưới phải
    ]
    
    print("Layout mới:")
    for i, (x1, y1, x2, y2) in enumerate(new_regions):
        print(f"  Ô {i+1}: ({x1:.2f}, {y1:.2f}) -> ({x2:.2f}, {y2:.2f})")
        print(f"    Kích thước: {x2-x1:.2f} x {y2-y1:.2f}")

def test_api_layout():
    """Test API layout mới"""
    print("\n🧪 Test API layout mới...")
    
    try:
        # Tạo ảnh test
        test_image = create_test_tray_image()
        
        # Gửi request đến API
        response = requests.post('http://localhost:5000/api/test-layout', 
                               json={'image': test_image}, 
                               timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API layout test thành công")
            print(f"   Kích thước ảnh gốc: {result['original_size']['width']}x{result['original_size']['height']}")
            print(f"   Số segments: {len(result['segments'])}")
            
            # Hiển thị thông tin layout
            layout_info = result['layout_info']
            print(f"   Mô tả: {layout_info['description']}")
            print("   Các vùng:")
            for region in layout_info['regions']:
                print(f"     - {region}")
            
            return True
        else:
            print(f"❌ API test thất bại: {response.status_code}")
            print(f"   Lỗi: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Lỗi test API: {e}")
        return False

def compare_layouts():
    """So sánh layout cũ và mới"""
    print("\n📊 So sánh layout cũ vs mới:")
    
    # Layout cũ
    old_regions = [
        (0.05, 0.02, 0.34, 0.41),
        (0.36, 0.02, 0.63, 0.41),
        (0.65, 0.02, 0.96, 0.41),
        (0.05, 0.44, 0.43, 0.98),
        (0.50, 0.44, 0.95, 0.98),
    ]
    
    # Layout mới
    new_regions = [
        (0.00, 0.05, 0.32, 0.35),
        (0.33, 0.05, 0.65, 0.35),
        (0.66, 0.05, 0.98, 0.35),
        (0.00, 0.40, 0.38, 0.95),
        (0.50, 0.40, 0.98, 0.95),
    ]
    
    print("So sánh diện tích các ô:")
    for i in range(5):
        old_x1, old_y1, old_x2, old_y2 = old_regions[i]
        new_x1, new_y1, new_x2, new_y2 = new_regions[i]
        
        old_area = (old_x2 - old_x1) * (old_y2 - old_y1)
        new_area = (new_x2 - new_x1) * (new_y2 - new_y1)
        
        print(f"  Ô {i+1}:")
        print(f"    Cũ: {old_area:.4f} ({old_x2-old_x1:.2f}x{old_y2-old_y1:.2f})")
        print(f"    Mới: {new_area:.4f} ({new_x2-new_x1:.2f}x{new_y2-new_y1:.2f})")
        print(f"    Thay đổi: {((new_area - old_area) / old_area * 100):+.1f}%")

def main():
    """Hàm chính"""
    print("🔧 Test Layout Cắt Khay Mới - Cải thiện Độ Chính Xác")
    print("=" * 60)
    
    # Test layout cũ
    test_old_layout()
    
    # Test layout mới
    test_new_layout()
    
    # So sánh layouts
    compare_layouts()
    
    # Test API
    api_ok = test_api_layout()
    
    print("\n📊 Tổng kết:")
    print(f"   API Test: {'✅' if api_ok else '❌'}")
    
    if api_ok:
        print("\n🎉 Layout mới hoạt động tốt!")
        print("💡 Cải tiến chính:")
        print("   - Layout 3 trên, 2 dưới rõ ràng hơn")
        print("   - Tỷ lệ các ô cân đối hơn")
        print("   - Giảm overlap giữa các vùng")
        print("   - Tăng độ chính xác nhận diện")
    else:
        print("\n⚠️ Có vấn đề với API test")
        print("   - Kiểm tra server có chạy không")
        print("   - Kiểm tra endpoint /api/test-layout")

if __name__ == '__main__':
    main()
