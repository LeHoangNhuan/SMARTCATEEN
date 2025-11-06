#!/usr/bin/env python3
"""
Script debug để kiểm tra và sửa lỗi ứng dụng nhận dạng thức ăn
"""

import requests
import json
import base64
from PIL import Image
import io
import sys
import os

def check_server_status():
    """Kiểm tra trạng thái server"""
    print("🔍 Kiểm tra trạng thái server...")
    try:
        response = requests.get('http://localhost:5000/api/health', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Server đang chạy")
            print(f"   Model đã tải: {data.get('model_loaded', False)}")
            print(f"   Số lớp thức ăn: {data.get('food_classes_count', 0)}")
            print(f"   Đường dẫn model: {data.get('model_path', 'N/A')}")
            return True
        else:
            print(f"❌ Server trả về lỗi: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Không thể kết nối đến server")
        print("   Vui lòng chạy: python app.py")
        return False
    except Exception as e:
        print(f"❌ Lỗi kiểm tra server: {e}")
        return False

def test_prediction_api():
    """Kiểm tra API dự đoán"""
    print("\n🧪 Kiểm tra API dự đoán...")
    
    # Tạo ảnh test đơn giản
    img = Image.new('RGB', (224, 224), color='red')
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    test_image = f"data:image/jpeg;base64,{img_str}"
    
    try:
        response = requests.post('http://localhost:5000/api/predict', 
                                json={'image': test_image}, 
                                timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API dự đoán hoạt động bình thường")
            print(f"   Kết quả: {result.get('prediction', {}).get('class', 'N/A')}")
            print(f"   Độ tin cậy: {result.get('prediction', {}).get('confidence', 0):.3f}")
            return True
        else:
            print(f"❌ API dự đoán lỗi: {response.status_code}")
            print(f"   Lỗi: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Lỗi test API: {e}")
        return False

def check_required_files():
    """Kiểm tra các file cần thiết"""
    print("\n📁 Kiểm tra file cần thiết...")
    
    required_files = [
        'app.py',
        'final_model.h5', 
        'cnn_classes.txt',
        'person_info.json',
        'nhan_dien.html'
    ]
    
    all_ok = True
    
    for file in required_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✅ {file} ({size} bytes)")
        else:
            print(f"❌ {file} - THIẾU")
            all_ok = False
    
    return all_ok

def check_dependencies():
    """Kiểm tra thư viện cần thiết"""
    print("\n📦 Kiểm tra thư viện...")
    
    required_packages = [
        'flask', 'flask_cors', 'tensorflow', 'numpy', 
        'PIL', 'cv2', 'requests'
    ]
    
    all_ok = True
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'cv2':
                import cv2
            elif package == 'flask_cors':
                import flask_cors
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - THIẾU")
            all_ok = False
    
    return all_ok

def test_browser_access():
    """Kiểm tra truy cập từ browser"""
    print("\n🌐 Kiểm tra truy cập web...")
    
    try:
        response = requests.get('http://localhost:5000/', timeout=5)
        if response.status_code == 200:
            print("✅ Trang web có thể truy cập")
            return True
        else:
            print(f"❌ Trang web lỗi: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Không thể truy cập trang web: {e}")
        return False

def main():
    """Hàm chính"""
    print("🔧 Debug Ứng dụng Nhận dạng Thức ăn AI")
    print("=" * 50)
    
    # Kiểm tra file
    files_ok = check_required_files()
    
    # Kiểm tra thư viện
    deps_ok = check_dependencies()
    
    # Kiểm tra server
    server_ok = check_server_status()
    
    # Kiểm tra web
    web_ok = test_browser_access()
    
    # Kiểm tra API
    api_ok = False
    if server_ok:
        api_ok = test_prediction_api()
    
    print("\n📊 Tổng kết:")
    print(f"   File cần thiết: {'✅' if files_ok else '❌'}")
    print(f"   Thư viện: {'✅' if deps_ok else '❌'}")
    print(f"   Server: {'✅' if server_ok else '❌'}")
    print(f"   Web: {'✅' if web_ok else '❌'}")
    print(f"   API: {'✅' if api_ok else '❌'}")
    
    if all([files_ok, deps_ok, server_ok, web_ok, api_ok]):
        print("\n🎉 Tất cả đều hoạt động bình thường!")
        print("💡 Nếu vẫn có lỗi, hãy:")
        print("   1. Mở Developer Tools (F12) trong browser")
        print("   2. Xem tab Console để kiểm tra lỗi JavaScript")
        print("   3. Xem tab Network để kiểm tra request")
    else:
        print("\n⚠️ Có vấn đề cần sửa:")
        if not files_ok:
            print("   - Kiểm tra các file cần thiết")
        if not deps_ok:
            print("   - Chạy: pip install -r requirements.txt")
        if not server_ok:
            print("   - Chạy: python app.py")
        if not web_ok:
            print("   - Kiểm tra server có chạy không")
        if not api_ok:
            print("   - Kiểm tra model và API")

if __name__ == '__main__':
    main()
