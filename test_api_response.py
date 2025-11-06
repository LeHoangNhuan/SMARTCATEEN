#!/usr/bin/env python3
"""
Script test để kiểm tra API response và thông tin thức ăn
"""

import requests
import json
import base64
from PIL import Image
import io

def create_test_image():
    """Tạo ảnh test"""
    img = Image.new('RGB', (224, 224), color='red')
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

def test_api_response():
    """Test API response"""
    print("🧪 Kiểm tra API response...")
    
    try:
        # Test health endpoint
        print("\n1. Kiểm tra health endpoint:")
        health_response = requests.get('http://localhost:5000/api/health')
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"   ✅ Model loaded: {health_data.get('model_loaded', False)}")
            print(f"   ✅ Food classes: {health_data.get('food_classes_count', 0)}")
        else:
            print(f"   ❌ Health check failed: {health_response.status_code}")
            return False
        
        # Test food info endpoint
        print("\n2. Kiểm tra food info endpoint:")
        info_response = requests.get('http://localhost:5000/api/food-info')
        if info_response.status_code == 200:
            info_data = info_response.json()
            print(f"   ✅ Food info items: {len(info_data)}")
            
            # Hiển thị một vài món ăn mẫu
            sample_foods = list(info_data.keys())[:3]
            for food in sample_foods:
                food_info = info_data[food]
                print(f"   📝 {food}:")
                print(f"      Giá: {food_info.get('Giá', 'N/A')}")
                print(f"      Calo: {food_info.get('Calo', 'N/A')}")
                print(f"      Loại: {food_info.get('Loại', 'N/A')}")
        else:
            print(f"   ❌ Food info failed: {info_response.status_code}")
        
        # Test prediction endpoint
        print("\n3. Kiểm tra prediction endpoint:")
        test_image = create_test_image()
        predict_response = requests.post('http://localhost:5000/api/predict', 
                                       json={'image': test_image})
        
        if predict_response.status_code == 200:
            predict_data = predict_response.json()
            print(f"   ✅ Prediction successful")
            print(f"   📊 Response structure:")
            print(f"      Success: {predict_data.get('success', False)}")
            
            if 'prediction' in predict_data:
                pred = predict_data['prediction']
                print(f"      Class: {pred.get('class', 'N/A')}")
                print(f"      Confidence: {pred.get('confidence', 0):.3f}")
                print(f"      Price: {pred.get('price', 'N/A')}")
                print(f"      Calories: {pred.get('calories', 'N/A')}")
                print(f"      Type: {pred.get('type', 'N/A')}")
                print(f"      Health Score: {pred.get('health_score', 'N/A')}")
                print(f"      Features: {pred.get('features', [])}")
            else:
                print("   ❌ No prediction data in response")
        else:
            print(f"   ❌ Prediction failed: {predict_response.status_code}")
            print(f"   Error: {predict_response.text}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi test API: {e}")
        return False

def test_food_info_mapping():
    """Kiểm tra mapping giữa tên món ăn và thông tin"""
    print("\n🔍 Kiểm tra mapping thông tin thức ăn...")
    
    try:
        # Lấy danh sách classes
        classes_response = requests.get('http://localhost:5000/api/classes')
        if classes_response.status_code != 200:
            print("❌ Không thể lấy danh sách classes")
            return False
        
        classes_data = classes_response.json()
        food_classes = classes_data.get('classes', [])
        
        # Lấy thông tin thức ăn
        info_response = requests.get('http://localhost:5000/api/food-info')
        if info_response.status_code != 200:
            print("❌ Không thể lấy thông tin thức ăn")
            return False
        
        food_info = info_response.json()
        
        print(f"📊 Tổng số classes: {len(food_classes)}")
        print(f"📊 Tổng số thông tin: {len(food_info)}")
        
        # Kiểm tra mapping
        missing_info = []
        for food_class in food_classes:
            if food_class not in food_info:
                missing_info.append(food_class)
                print(f"❌ Thiếu thông tin cho: {food_class}")
            else:
                info = food_info[food_class]
                print(f"✅ {food_class}: {info.get('Giá', 'N/A')} - {info.get('Calo', 'N/A')}")
        
        if missing_info:
            print(f"\n⚠️ Thiếu thông tin cho {len(missing_info)} món ăn:")
            for food in missing_info:
                print(f"   - {food}")
        else:
            print("\n✅ Tất cả món ăn đều có thông tin đầy đủ")
        
        return len(missing_info) == 0
        
    except Exception as e:
        print(f"❌ Lỗi kiểm tra mapping: {e}")
        return False

def main():
    """Hàm chính"""
    print("🔍 Kiểm tra API Response và Thông tin Thức ăn")
    print("=" * 60)
    
    # Test API response
    api_ok = test_api_response()
    
    # Test food info mapping
    mapping_ok = test_food_info_mapping()
    
    print("\n📊 Tổng kết:")
    print(f"   API Response: {'✅' if api_ok else '❌'}")
    print(f"   Food Info Mapping: {'✅' if mapping_ok else '❌'}")
    
    if api_ok and mapping_ok:
        print("\n🎉 Tất cả đều hoạt động bình thường!")
        print("💡 Nếu vẫn không hiển thị thông tin, hãy:")
        print("   1. Mở Developer Tools (F12) trong browser")
        print("   2. Xem tab Console để kiểm tra log")
        print("   3. Xem tab Network để kiểm tra API calls")
    else:
        print("\n⚠️ Có vấn đề cần sửa:")
        if not api_ok:
            print("   - Kiểm tra server có chạy không")
            print("   - Kiểm tra model có load không")
        if not mapping_ok:
            print("   - Kiểm tra file person_info.json")
            print("   - Kiểm tra tên món ăn có khớp không")

if __name__ == '__main__':
    main()
