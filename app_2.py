from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import json
import os
import cv2

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
# ========== THAY ĐỔI: Sử dụng model của Code 2 ==========
MODEL_PATH = 'final_model.h5'  # Model từ Code 2
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and food information
model = None
_loaded_model_path = None

# ========== Danh sách classes  ==========
food_classes = [
    'Canh chua có cá',
    'Canh chua không cá', 
    'Canh rau cải',
    'Canh rau muống',
    'Cá hú kho',
    'Cơm trắng',
    'Củ sắn xào',
    'Khay trống',
    'Lagim xào',
    'Sườn nướng',
    'Thịt kho',
    'Thịt kho 1 trứng',
    'Thịt kho 2 trứng',
    'Trứng chiên',
    'Đậu hũ sốt cà',
    'Đậu que xào',
    'Đậu đũa xào'
]

# ========== THAY ĐỔI: Load food info từ Code 2 (hoặc giữ person_info.json) ==========

food_info = {
    'Canh chua có cá': {'Giá': 25000, 'Calo': 80, 'Loại': 'Canh', 'HealthyScore': 7, 'Đặc điểm': ['Chua', 'Cá']},
    'Canh chua không cá': {'Giá': 10000, 'Calo': 70, 'Loại': 'Canh', 'HealthyScore': 8, 'Đặc điểm': ['Chua', 'Chay']},
    'Canh rau cải': {'Giá': 7000, 'Calo': 50, 'Loại': 'Canh', 'HealthyScore': 9, 'Đặc điểm': ['Rau xanh']},
    'Canh rau muống': {'Giá': 7000, 'Calo': 40, 'Loại': 'Canh', 'HealthyScore': 9, 'Đặc điểm': ['Rau xanh']},
    'Cá hú kho': {'Giá': 30000, 'Calo': 150, 'Loại': 'Món mặn', 'HealthyScore': 7, 'Đặc điểm': ['Cá', 'Kho']},
    'Cơm trắng': {'Giá': 10000, 'Calo': 250, 'Loại': 'Món chính', 'HealthyScore': 6, 'Đặc điểm': ['Tinh bột']},
    'Củ sắn xào': {'Giá': 10000, 'Calo': 120, 'Loại': 'Rau', 'HealthyScore': 7, 'Đặc điểm': ['Củ', 'Xào']},
    'Khay trống': {'Giá': 0, 'Calo': 0, 'Loại': 'Trống', 'HealthyScore': 0, 'Đặc điểm': []},
    'Lagim xào': {'Giá': 10000, 'Calo': 120, 'Loại': 'Rau', 'HealthyScore': 8, 'Đặc điểm': ['Rau xanh', 'Xào']},
    'Sường nướng': {'Giá': 30000, 'Calo': 300, 'Loại': 'Món mặn', 'HealthyScore': 5, 'Đặc điểm': ['Thịt', 'Nướng']},
    'Thịt kho': {'Giá': 25000, 'Calo': 250, 'Loại': 'Món mặn', 'HealthyScore': 5, 'Đặc điểm': ['Thịt', 'Kho']},
    'Thịt kho 1 trứng': {'Giá': 30000, 'Calo': 300, 'Loại': 'Món mặn', 'HealthyScore': 5, 'Đặc điểm': ['Thịt', 'Trứng', 'Kho']},
    'Thịt kho 2 trứng': {'Giá': 36000, 'Calo': 350, 'Loại': 'Món mặn', 'HealthyScore': 5, 'Đặc điểm': ['Thịt', 'Trứng', 'Kho']},
    'Trứng chiên': {'Giá': 25000, 'Calo': 150, 'Loại': 'Món mặn', 'HealthyScore': 6, 'Đặc điểm': ['Trứng', 'Chiên']},
    'Đậu hũ sốt cà': {'Giá': 25000, 'Calo': 100, 'Loại': 'Món chay', 'HealthyScore': 7, 'Đặc điểm': ['Đậu', 'Sốt cà']},
    'Đậu que xào': {'Giá': 10000, 'Calo': 80, 'Loại': 'Rau', 'HealthyScore': 8, 'Đặc điểm': ['Đậu', 'Xào']},
    'Đậu đũa xào': {'Giá': 10000, 'Calo': 90, 'Loại': 'Rau', 'HealthyScore': 8, 'Đặc điểm': ['Đậu', 'Xào']}
}
# ==================== Model utilities ====================

def _resolve_model_path() -> str:
    """Get the ultimate model path."""
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH
    return None


def load_model():
    """Load the trained model"""
    global model, _loaded_model_path
    try:
        model_path = _resolve_model_path()
        if not model_path:
            print(f"No model file found: {MODEL_PATH}")
            return False
        
        # ========== THAY ĐỔI: Load model H5 thay vì Keras ==========
        model = tf.keras.models.load_model(model_path, compile=False)
        print(f"Model loaded successfully from {model_path}")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        _loaded_model_path = model_path
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


def _get_model_num_classes() -> int:
    try:
        if model is None:
            return None
        out_shape = getattr(model, 'output_shape', None)
        if isinstance(out_shape, (list, tuple)) and len(out_shape) >= 2:
            return int(out_shape[-1]) if out_shape and isinstance(out_shape[-1], int) else None
        return None
    except Exception:
        return None


def _get_model_input_size() -> tuple:
    try:
        if model is None:
            return None
        in_shape = getattr(model, 'input_shape', None)
        # Expect (None, H, W, C)
        if isinstance(in_shape, (list, tuple)) and len(in_shape) >= 4:
            h, w = int(in_shape[1]), int(in_shape[2])
            return (w, h)
        return None
    except Exception:
        return None

# ==================== Tray utilities ====================
def _decode_base64_to_cv2(image_data: str):
    try:
        if isinstance(image_data, str) and image_data.startswith('data:'):
            image_data = image_data.split(',', 1)[1]
        image_bytes = base64.b64decode(image_data)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return img_bgr
    except Exception:
        return None


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _get_five_compartment_boxes(width: int, height: int, padding_ratio: float = 0.02):
    grid = [
        (0.14, 0.04, 0.38, 0.41),
        (0.35, 0.04, 0.58, 0.41),
        (0.55, 0.04, 0.78, 0.41),
        (0.16, 0.42, 0.41, 0.92),
        (0.45, 0.42, 0.78, 0.92),
    ]
    boxes = []
    px = padding_ratio * width
    py = padding_ratio * height
    for (x1r, y1r, x2r, y2r) in grid:
        x1 = int(_clamp(x1r * width + px, 0, width - 1))
        y1 = int(_clamp(y1r * height + py, 0, height - 1))
        x2 = int(_clamp(x2r * width - px, 0, width - 1))
        y2 = int(_clamp(y2r * height - py, 0, height - 1))
        if x2 <= x1:
            x2 = min(width - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(height - 1, y1 + 1)
        boxes.append((x1, y1, x2, y2))
    return boxes


def _preprocess_cv_crop_for_model(crop_bgr: np.ndarray) -> np.ndarray:
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    try:
        input_size = _get_model_input_size() or (224, 224)
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, input_size, interpolation=cv2.INTER_AREA)
        arr = resized.astype(np.float32) / 255.0
        return arr
    except Exception:
        return None


def _apply_position_priors_to_scores(scores: np.ndarray, slot_index: int) -> np.ndarray:
    priors = _build_position_priors()
    rice_idx = priors.get('rice_idx')
    soup_indices = priors.get('soup_indices', set())
    adjusted = scores.copy()
    if slot_index in (0, 1, 2):
        for si in soup_indices:
            if 0 <= si < adjusted.shape[0]:
                adjusted[si] *= 1.10
    if slot_index in (3, 4) and rice_idx is not None and 0 <= rice_idx < adjusted.shape[0]:
        adjusted[rice_idx] *= 1.15
    s = adjusted.sum()
    if s > 0:
        adjusted = adjusted / s
    return adjusted


def _find_empty_class_index() -> int:
    for idx, name in enumerate(food_classes):
        n = (name or '').lower()
        # ========== THAY ĐỔI: "Khay trống" thay vì "Không có món" ==========
        if 'khay trống' in n or 'khay trong' in n or 'empty' in n:
            return idx
    return -1


def _predict_tray_slots(
    image_data: str,
    padding_ratio: float = 0.05,
    min_confidence: float = 0.6,
    empty_margin: float = 0.1,
    empty_var_thresh: float = 12.0,
    empty_sat_thresh: float = 15.0,
):
    if model is None:
        return {'error': 'Model not loaded'}
    img = _decode_base64_to_cv2(image_data)
    if img is None:
        return {'error': 'Invalid image data'}
    h, w = img.shape[:2]
    boxes = _get_five_compartment_boxes(w, h, padding_ratio=padding_ratio)

    crops = []
    raw_crops_bgr = []
    empty_hints = {}
    yellow_ratios = {}
    red_ratios = {}
    valid_indices = []
    
    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        crop = img[y1:y2, x1:x2]
        try:
            if crop is not None and crop.size > 0:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                _, stddev = cv2.meanStdDev(gray)
                std_val = float(stddev[0][0])
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                sat_mean = float(cv2.mean(hsv[:, :, 1])[0])
                empty_hints[idx] = (std_val < empty_var_thresh) and (sat_mean < empty_sat_thresh)
                
                # Yellow detection (for eggs and Thịt kho trứng)
                try:
                    lower1 = np.array([8, 50, 80], dtype=np.uint8)
                    upper1 = np.array([55, 255, 255], dtype=np.uint8)
                    mask1 = cv2.inRange(hsv, lower1, upper1)
                    
                    lower2 = np.array([12, 80, 120], dtype=np.uint8)
                    upper2 = np.array([40, 255, 255], dtype=np.uint8)
                    mask2 = cv2.inRange(hsv, lower2, upper2)
                    
                    lower3 = np.array([15, 30, 200], dtype=np.uint8)
                    upper3 = np.array([35, 100, 255], dtype=np.uint8)
                    mask3 = cv2.inRange(hsv, lower3, upper3)
                    
                    combined_mask = mask1 | mask2 | mask3
                    ratio = float(cv2.countNonZero(combined_mask)) / float(combined_mask.size)
                    yellow_ratios[idx] = ratio
                except Exception:
                    yellow_ratios[idx] = 0.0
                    
                # Red detection (for meat dishes)
                try:
                    lower1 = np.array([0, 90, 110], dtype=np.uint8)
                    upper1 = np.array([10, 255, 255], dtype=np.uint8)
                    lower2 = np.array([170, 90, 110], dtype=np.uint8)
                    upper2 = np.array([179, 255, 255], dtype=np.uint8)
                    mask1 = cv2.inRange(hsv, lower1, upper1)
                    mask2 = cv2.inRange(hsv, lower2, upper2)
                    rratio = float(cv2.countNonZero(mask1 | mask2)) / float(hsv.shape[0] * hsv.shape[1])
                    red_ratios[idx] = rratio
                except Exception:
                    red_ratios[idx] = 0.0
            else:
                empty_hints[idx] = False
                yellow_ratios[idx] = 0.0
                red_ratios[idx] = 0.0
        except Exception:
            empty_hints[idx] = False
            yellow_ratios[idx] = 0.0
            red_ratios[idx] = 0.0
            
        arr = _preprocess_cv_crop_for_model(crop)
        if arr is not None:
            crops.append(arr)
            raw_crops_bgr.append(crop)
            valid_indices.append(idx)

    if not crops:
        return {'error': 'Failed to prepare tray crops'}

    batch = np.stack(crops, axis=0)
    preds = model.predict(batch, verbose=0)
    slot_results = [None] * len(boxes)
    empty_idx = _find_empty_class_index()
    adjusted_scores_per_slot = {}
    raw_scores_per_slot = {}
    
    for i, slot_idx in enumerate(valid_indices):
        scores = preds[i]
        scores = _apply_position_priors_to_scores(scores, slot_idx)
        
        # ========== Color-based heuristics (LOGIC HỢP LÝ) ==========
        try:
            yellow_ratio = float(yellow_ratios.get(slot_idx, 0.0))
            red_ratio = float(red_ratios.get(slot_idx, 0.0))
            
            # Tìm indices
            tkt1_idx, tkt2_idx, egg_idx, thit_kho_idx = None, None, None, None
            for ci, name in enumerate(food_classes):
                n = (name or '').lower()
                if tkt1_idx is None and 'thịt kho 1 trứng' in n:
                    tkt1_idx = ci
                if tkt2_idx is None and 'thịt kho 2 trứng' in n:
                    tkt2_idx = ci
                if egg_idx is None and 'trứng chiên' in n:
                    egg_idx = ci
                if thit_kho_idx is None and 'thịt kho' == n.strip():
                    thit_kho_idx = ci
            
            scores = scores.copy()
            
            # LOGIC 1: Nhiều vàng + ÍT đỏ = Trứng chiên (lòng đỏ lộ rõ)
            if yellow_ratio > 0.08 and red_ratio < 0.05:
                if egg_idx is not None and 0 <= egg_idx < scores.shape[0]:
                    scores[egg_idx] *= 2.5  # Boost mạnh cho trứng chiên
                # Giảm Thịt kho trứng vì không có màu đỏ nước kho
                if tkt1_idx is not None and 0 <= tkt1_idx < scores.shape[0]:
                    scores[tkt1_idx] *= 0.3
                if tkt2_idx is not None and 0 <= tkt2_idx < scores.shape[0]:
                    scores[tkt2_idx] *= 0.3
            
            # LOGIC 2: Nhiều đỏ + ÍT vàng = Thịt kho (không có trứng) hoặc Thịt kho trứng (trứng nguyên quả)
            elif red_ratio > 0.1:
                # Nếu có một chút vàng (có thể là trứng nguyên quả bị vỡ) → boost Thịt kho trứng
                if yellow_ratio > 0.02 and yellow_ratio < 0.08:
                    if tkt1_idx is not None and 0 <= tkt1_idx < scores.shape[0]:
                        scores[tkt1_idx] *= 1.5
                    if tkt2_idx is not None and 0 <= tkt2_idx < scores.shape[0]:
                        scores[tkt2_idx] *= 1.6
                    # Giảm Trứng chiên vì có nhiều đỏ (không phải trứng chiên)
                    if egg_idx is not None and 0 <= egg_idx < scores.shape[0]:
                        scores[egg_idx] *= 0.4
                # Nếu KHÔNG có vàng → boost Thịt kho (không trứng)
                elif yellow_ratio <= 0.02:
                    if thit_kho_idx is not None and 0 <= thit_kho_idx < scores.shape[0]:
                        scores[thit_kho_idx] *= 1.8
                    # Giảm cả Trứng chiên và Thịt kho trứng
                    if egg_idx is not None and 0 <= egg_idx < scores.shape[0]:
                        scores[egg_idx] *= 0.2
                    if tkt1_idx is not None and 0 <= tkt1_idx < scores.shape[0]:
                        scores[tkt1_idx] *= 0.5
                    if tkt2_idx is not None and 0 <= tkt2_idx < scores.shape[0]:
                        scores[tkt2_idx] *= 0.5
            
            # Renormalize
            s = scores.sum()
            if s > 0:
                scores = scores / s
        except Exception:
            pass
            
        raw_scores_per_slot[slot_idx] = preds[i]
        adjusted_scores_per_slot[slot_idx] = scores
        max_index = int(np.argmax(scores))
        confidence = float(scores[max_index])
        
        if empty_idx >= 0 and confidence < float(min_confidence):
            empty_prob = float(scores[empty_idx]) if empty_idx < scores.shape[0] else 0.0
            if empty_prob >= confidence - float(empty_margin):
                max_index = empty_idx
                confidence = empty_prob
                
        if empty_idx >= 0 and empty_hints.get(slot_idx, False):
            max_index = empty_idx
            confidence = max(confidence, float(scores[empty_idx]) if empty_idx < scores.shape[0] else 0.95)
            
        try:
            top_indices = np.argsort(scores)[-3:][::-1]
            top3 = [
                {
                    'class': (food_classes[int(ci)] if int(ci) < len(food_classes) else str(int(ci))),
                    'confidence': float(scores[int(ci)])
                }
                for ci in top_indices
            ]
        except Exception:
            top3 = []
        
        debug_info = {
            'yellow_ratio': float(yellow_ratios.get(slot_idx, 0.0)),
            'red_ratio': float(red_ratios.get(slot_idx, 0.0)),
            'empty_hint': empty_hints.get(slot_idx, False)
        }
        
        label = food_classes[max_index] if max_index < len(food_classes) else str(max_index)
        slot_results[slot_idx] = {
            'slot': slot_idx + 1,
            'class': label,
            'confidence': confidence,
            'box': boxes[slot_idx],
            'top3': top3,
            'debug': debug_info,
        }

    # ========== Global optimization để tránh duplicate ==========
    try:
        top_k = 3
        num_slots = len(valid_indices)
        slot_to_candidates = {}
        for slot_idx in valid_indices:
            scores = adjusted_scores_per_slot[slot_idx]
            top_indices = np.argsort(scores)[-top_k:][::-1]
            slot_to_candidates[slot_idx] = [(int(ci), float(scores[ci])) for ci in top_indices]

        pri = _build_position_priors()
        rice_idx = pri.get('rice_idx')
        soup_set = pri.get('soup_indices', set())

        def pos_penalty(slot_idx: int, cls_idx: int) -> float:
            if slot_idx in (0, 1, 2) and cls_idx in soup_set:
                return 0.0
            if slot_idx in (3, 4) and rice_idx is not None and cls_idx == rice_idx:
                return 0.0
            return 0.2

        slots_order = sorted(valid_indices)
        best = {'cost': float('inf'), 'choice': None}

        def dfs(idx: int, choice: list, used_counts: dict, total_cost: float):
            if idx == len(slots_order):
                if total_cost < best['cost']:
                    best['cost'] = total_cost
                    best['choice'] = list(choice)
                return
            slot_idx = slots_order[idx]
            for cls_idx, prob in slot_to_candidates[slot_idx]:
                p = max(prob, 1e-6)
                cost = -float(np.log(p)) + pos_penalty(slot_idx, cls_idx)
                cnt = used_counts.get(cls_idx, 0)
                if cnt >= 1:
                    cost += 0.7 * cnt
                new_total = total_cost + cost
                if new_total >= best['cost']:
                    continue
                used_counts[cls_idx] = cnt + 1
                choice.append((slot_idx, cls_idx, prob))
                dfs(idx + 1, choice, used_counts, new_total)
                choice.pop()
                if cnt == 0:
                    del used_counts[cls_idx]
                else:
                    used_counts[cls_idx] = cnt

        dfs(0, [], {}, 0.0)

        if best['choice']:
            for slot_idx, cls_idx, prob in best['choice']:
                label = food_classes[cls_idx] if cls_idx < len(food_classes) else str(cls_idx)
                prev = slot_results[slot_idx] or {}
                slot_results[slot_idx] = {
                    'slot': slot_idx + 1,
                    'class': label,
                    'confidence': float(prob),
                    'box': boxes[slot_idx],
                    'top3': prev.get('top3', []),
                }
    except Exception as _:
        pass

    return {
        'slots': slot_results,
        'image_size': {'width': w, 'height': h},
    }

# ========== Position priors helper ==========
_PRIOR_CACHE = None
def _build_position_priors():
    global _PRIOR_CACHE
    if _PRIOR_CACHE is not None:
        return _PRIOR_CACHE
    rice_idx = None
    soup_indices = []
    try:
        for idx, name in enumerate(food_classes):
            n = (name or '').lower()
            if 'cơm trắng' in n or 'com trắng' in n or 'com trang' in n or 'cơm trang' in n:
                rice_idx = idx
            if 'canh' in n:
                soup_indices.append(idx)
    except Exception:
        pass
    _PRIOR_CACHE = {
        'rice_idx': rice_idx,
        'soup_indices': set(soup_indices),
    }
    return _PRIOR_CACHE

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_data):
    try:
        if isinstance(image_data, str) and image_data.startswith('data:'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_food(image_array):
    try:
        if model is None:
            return None, 0.0
        
        predictions = model.predict(image_array, verbose=0)
        max_index = np.argmax(predictions[0])
        confidence = float(predictions[0][max_index])
        
        if max_index >= len(food_classes):
            print(f"Warning: predicted index {max_index} out of range for labels {len(food_classes)}")
            predicted_class = str(max_index)
        else:
            predicted_class = food_classes[max_index]
        
        return predicted_class, confidence
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None, 0.0


# ==================== ROUTES ====================
@app.route('/')
def index():
    return send_from_directory('.', 'nhan_dien.html')

@app.route('/original')
def original():
    return send_from_directory('.', 'menu.html')

@app.route('/gioithieu')
def gioithieu():
    return send_from_directory('.', 'gioithieu.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)

@app.route('/model/<path:filename>')
def serve_model(filename):
    return send_from_directory('.', filename)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if request.files:
            file = request.files.get('image')
            if not file:
                return jsonify({'error': 'No image file provided'}), 400
            
            image = Image.open(file.stream)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image = image.resize((224, 224))
            image_array = np.array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)
            
        else:
            data = request.get_json()
            if not data or 'image' not in data:
                return jsonify({'error': 'No image data provided'}), 400
            
            image_array = preprocess_image(data['image'])
            if image_array is None:
                return jsonify({'error': 'Invalid image data'}), 400
        
        predicted_class, confidence = predict_food(image_array)
        
        if predicted_class is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        food_data = food_info.get(predicted_class, {})
        
        response = {
            'success': True,
            'prediction': {
                'class': predicted_class,
                'confidence': confidence,
                'price': food_data.get('Giá', 'N/A'),
                'calories': food_data.get('Calo', 'N/A'),
                'type': food_data.get('Loại', 'Món ăn'),
                'health_score': food_data.get('HealthyScore', 5),
                'features': food_data.get('Đặc điểm', [])
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in predict endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/health')
def health_check():
    model_path = None
    try:
        model_path = _loaded_model_path or _resolve_model_path()
    except Exception:
        model_path = None
    
    model_num_classes = _get_model_num_classes()
    labels_mismatch = (model_num_classes is not None and model_num_classes != len(food_classes))
    model_input_size = _get_model_input_size()
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_path': model_path,
        'food_classes_count': len(food_classes),
        'model_num_classes': model_num_classes,
        'labels_mismatch': labels_mismatch,
        'model_input_size': model_input_size,
    })

@app.route('/api/food-info')
def get_food_info():
    return jsonify(food_info)

@app.route('/api/classes')
def get_classes():
    return jsonify(food_classes)


@app.route('/api/predict-tray', methods=['POST'])
def predict_tray():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        padding_ratio = float(data.get('padding_ratio', 0.02))
        min_conf = float(data.get('min_confidence', 0.6))
        empty_margin = float(data.get('empty_margin', 0.1))
        empty_var_thresh = float(data.get('empty_var_thresh', 12.0))
        empty_sat_thresh = float(data.get('empty_sat_thresh', 15.0))
        result = _predict_tray_slots(
            data['image'], padding_ratio=padding_ratio, min_confidence=min_conf, empty_margin=empty_margin,
            empty_var_thresh=empty_var_thresh, empty_sat_thresh=empty_sat_thresh
        )
        if 'error' in result:
            return jsonify(result), 400
        
        enriched = []
        for item in result['slots']:
            if not item:
                enriched.append(None)
                continue
            info = food_info.get(item['class'], {})
            enriched.append({
                **item,
                'price': info.get('Giá', 'N/A'),
                'calories': info.get('Calo', 'N/A'),
                'type': info.get('Loại', 'Món ăn'),
                'health_score': info.get('HealthyScore', 5),
                'features': info.get('Đặc điểm', []),
            })
        return jsonify({
            'success': True,
            'image_size': result['image_size'],
            'predictions': enriched,
        })
    except Exception as e:
        print(f"Error in predict-tray endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/crop-image', methods=['POST'])
def crop_image():
    try:
        if request.files:
            file = request.files.get('image')
            if not file:
                return jsonify({'error': 'No image file provided'}), 400
            
            image = Image.open(file.stream)
            if image.mode != 'RGB':
                image = image.convert('RGB')
        else:
            data = request.get_json()
            if not data or 'image' not in data:
                return jsonify({'error': 'No image data provided'}), 400
            
            try:
                image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
                if image.mode != 'RGB':
                    image = image.convert('RGB')
            except Exception as e:
                return jsonify({'error': f'Invalid image data: {str(e)}'}), 400
        
        width, height = image.size
        
        crop_regions = [
            (int(0.05 * width), int(0.02 * height), int(0.34 * width), int(0.41 * height)),
            (int(0.36 * width), int(0.02 * height), int(0.63 * width), int(0.41 * height)),
            (int(0.65 * width), int(0.02 * height), int(0.96 * width), int(0.41 * height)),
            (int(0.05 * width), int(0.44 * height), int(0.43 * width), int(0.98 * height)),
            (int(0.50 * width), int(0.44 * height), int(0.95 * width), int(0.98 * height)),
        ]
        
        cropped_images = []
        predictions = []
        
        for i, (left, top, right, bottom) in enumerate(crop_regions):
            cropped = image.crop((left, top, right, bottom))
            cropped_resized = cropped.resize((224, 224))
            cropped_array = np.array(cropped_resized) / 255.0
            cropped_array = np.expand_dims(cropped_array, axis=0)
            
            predicted_class, confidence = predict_food(cropped_array)
            food_data = food_info.get(predicted_class, {}) if predicted_class else {}
            
            buffered = io.BytesIO()
            cropped_resized.save(buffered, format="JPEG")
            cropped_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            cropped_images.append(f"data:image/jpeg;base64,{cropped_base64}")
            
            predictions.append({
                'region': i + 1,
                'class': predicted_class,
                'confidence': confidence,
                'price': food_data.get('Giá', 'N/A'),
                'calories': food_data.get('Calo', 'N/A'),
                'type': food_data.get('Loại', 'Món ăn'),
                'health_score': food_data.get('HealthyScore', 5),
                'features': food_data.get('Đặc điểm', [])
            })
        
        return jsonify({
            'success': True,
            'cropped_images': cropped_images,
            'predictions': predictions,
            'original_size': {'width': width, 'height': height}
        })
        
    except Exception as e:
        print(f"Error in crop_image endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500



if __name__ == '__main__':
    print("=" * 60)
    print("🚀 STARTING HYBRID AI FOOD RECOGNITION SERVER")
    print("=" * 60)
    print("📌 Using Code 2 Model + Code 1 Logic")
    print("=" * 60)
    
    # Load model
    if load_model():
        print("✅ Model loaded successfully!")
        print(f"   Path: {MODEL_PATH}")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
    else:
        print("❌ Failed to load model!")
        print(f"   Expected path: {MODEL_PATH}")
        exit(1)
    
    print(f"\n📊 Food Classes: {len(food_classes)}")
    print(f"   Classes: {', '.join(food_classes[:5])}...")
    
    print(f"\n🍽️  Food Info Database: {len(food_info)} items")
    
    print("\n🎯 Features:")
    print("   ✓ Single dish prediction (/api/predict)")
    print("   ✓ 5-compartment tray analysis (/api/predict-tray)")
    print("   ✓ Color-based heuristics (yellow for eggs, red for meat)")
    print("   ✓ Position priors (soups on top, rice on bottom)")
    print("   ✓ Global optimization (avoid duplicates)")
    print("   ✓ Empty compartment detection")
    
    print("\n" + "=" * 60)
    print("🌐 Server starting on http://localhost:5000")
    print("=" * 60)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )