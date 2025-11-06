#!/usr/bin/env python3
"""
Script kiểm tra đơn giản để xác minh chức năng của app.py
"""

import requests
import json
import base64
from PIL import Image
import io

def test_health_endpoint():
    """Kiểm tra endpoint kiểm tra sức khỏe"""
    try:
        response = requests.get('http://localhost:5000/api/health')
        if response.status_code == 200:
            data = response.json()
            print("✅ Kiểm tra sức khỏe thành công")
            print(f"   Model đã tải: {data.get('model_loaded', False)}")
            print(f"   Số lượng lớp thức ăn: {data.get('food_classes_count', 0)}")
            return True
        else:
            print(f"❌ Kiểm tra sức khỏe thất bại: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Lỗi kiểm tra sức khỏe: {e}")
        return False

def test_classes_endpoint():
    """Kiểm tra endpoint lớp"""
    try:
        response = requests.get('http://localhost:5000/api/classes')
        if response.status_code == 200:
            data = response.json()
            print("✅ Endpoint lớp hoạt động bình thường")
            print(f"   Số lượng lớp: {data.get('count', 0)}")
            print(f"   3 lớp đầu tiên: {data.get('classes', [])[:3]}")
            return True
        else:
            print(f"❌ Endpoint lớp thất bại: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Lỗi endpoint lớp: {e}")
        return False

def test_food_info_endpoint():
    """Kiểm tra endpoint thông tin thức ăn"""
    try:
        response = requests.get('http://localhost:5000/api/food-info')
        if response.status_code == 200:
            data = response.json()
            print("✅ Endpoint thông tin thức ăn hoạt động bình thường")
            print(f"   Số lượng thông tin thức ăn: {len(data)}")
            return True
        else:
            print(f"❌ Endpoint thông tin thức ăn thất bại: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Lỗi endpoint thông tin thức ăn: {e}")
        return False

def create_test_image():
    """Tạo một hình ảnh kiểm tra"""
    # Tạo một hình ảnh kiểm tra đơn giản
    img = Image.new('RGB', (224, 224), color='red')
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

def test_prediction_endpoint():
    """Kiểm tra endpoint dự đoán"""
    try:
        test_image = create_test_image()
        data = {'image': test_image}
        response = requests.post('http://localhost:5000/api/predict', json=data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Endpoint dự đoán hoạt động bình thường")
            print(f"   Kết quả dự đoán: {result.get('prediction', {}).get('class', 'N/A')}")
            print(f"   Độ tin cậy: {result.get('prediction', {}).get('confidence', 0):.3f}")
            return True
        else:
            print(f"❌ Endpoint dự đoán thất bại: {response.status_code}")
            print(f"   Thông báo lỗi: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Lỗi endpoint dự đoán: {e}")
        return False

def main():
    """Chạy tất cả các kiểm tra"""
    print("🧪 Bắt đầu kiểm tra ứng dụng nhận dạng thức ăn AI...")
    print("=" * 50)
    
    tests = [
        ("Kiểm tra sức khỏe", test_health_endpoint),
        ("Endpoint lớp", test_classes_endpoint),
        ("Endpoint thông tin thức ăn", test_food_info_endpoint),
        ("Endpoint dự đoán", test_prediction_endpoint),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Kiểm tra: {test_name}")
        if test_func():
            passed += 1
        print("-" * 30)
    
    print(f"\n📊 Kết quả kiểm tra: {passed}/{total} thành công")
    
    if passed == total:
        print("🎉 Tất cả kiểm tra đều thành công! Ứng dụng hoạt động bình thường.")
    else:
        print("⚠️ Một số kiểm tra thất bại, vui lòng kiểm tra trạng thái ứng dụng.")

if __name__ == '__main__':
    main()
