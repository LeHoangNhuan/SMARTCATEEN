#!/usr/bin/env python3
"""
Script khởi động ứng dụng Nhận dạng Thức ăn AI
"""

import os
import sys
import subprocess
import time

def check_python_version():
    """Kiểm tra phiên bản Python"""
    if sys.version_info < (3, 7):
        print("❌ Yêu cầu Python 3.7 trở lên!")
        print(f"   Phiên bản hiện tại: {sys.version}")
        return False
    print(f"✅ Python {sys.version.split()[0]} - OK")
    return True

def check_dependencies():
    """Kiểm tra các thư viện cần thiết"""
    required_packages = [
        'flask', 'flask_cors', 'tensorflow', 'numpy', 
        'PIL', 'cv2', 'requests'
    ]
    
    missing_packages = []
    
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
            print(f"✅ {package} - OK")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - Thiếu")
    
    if missing_packages:
        print(f"\n⚠️ Thiếu các thư viện: {', '.join(missing_packages)}")
        print("Chạy lệnh sau để cài đặt:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def check_required_files():
    """Kiểm tra các file cần thiết"""
    required_files = [
        'app.py',
        'final_model.h5',
        'cnn_classes.txt', 
        'person_info.json'
    ]
    
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} - OK")
        else:
            missing_files.append(file)
            print(f"❌ {file} - Thiếu")
    
    if missing_files:
        print(f"\n⚠️ Thiếu các file: {', '.join(missing_files)}")
        return False
    
    return True

def create_upload_folder():
    """Tạo thư mục uploads nếu chưa có"""
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
        print("✅ Tạo thư mục uploads")
    else:
        print("✅ Thư mục uploads đã tồn tại")

def start_application():
    """Khởi động ứng dụng"""
    print("\n🚀 Đang khởi động ứng dụng...")
    print("=" * 50)
    
    try:
        # Import và chạy app
        import app
        print("✅ Ứng dụng đã khởi động thành công!")
        print("🌐 Truy cập: http://localhost:5000")
        print("📊 API Health: http://localhost:5000/api/health")
        print("\n💡 Nhấn Ctrl+C để dừng ứng dụng")
        
    except Exception as e:
        print(f"❌ Lỗi khởi động ứng dụng: {e}")
        return False
    
    return True

def main():
    """Hàm chính"""
    print("🍽️ Ứng dụng Nhận dạng Thức ăn AI")
    print("=" * 50)
    
    # Kiểm tra Python version
    if not check_python_version():
        return False
    
    print("\n📦 Kiểm tra thư viện...")
    if not check_dependencies():
        return False
    
    print("\n📁 Kiểm tra file cần thiết...")
    if not check_required_files():
        return False
    
    print("\n📂 Kiểm tra thư mục...")
    create_upload_folder()
    
    print("\n✅ Tất cả kiểm tra đều thành công!")
    
    # Hỏi người dùng có muốn chạy test không
    try:
        run_test = input("\n🧪 Bạn có muốn chạy test trước khi khởi động? (y/n): ").lower().strip()
        if run_test in ['y', 'yes', 'có']:
            print("\n🔍 Chạy test...")
            try:
                result = subprocess.run([sys.executable, 'test_app.py'], 
                                      capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    print("✅ Test thành công!")
                else:
                    print("⚠️ Test có vấn đề, nhưng vẫn tiếp tục khởi động...")
                    print(f"Lỗi: {result.stderr}")
            except subprocess.TimeoutExpired:
                print("⚠️ Test timeout, tiếp tục khởi động...")
            except Exception as e:
                print(f"⚠️ Lỗi test: {e}, tiếp tục khởi động...")
    except KeyboardInterrupt:
        print("\n👋 Tạm biệt!")
        return False
    
    # Khởi động ứng dụng
    return start_application()

if __name__ == '__main__':
    try:
        success = main()
        if not success:
            print("\n❌ Khởi động thất bại!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n👋 Tạm biệt!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Lỗi không mong muốn: {e}")
        sys.exit(1)
