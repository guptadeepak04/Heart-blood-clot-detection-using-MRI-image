from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import base64
import json
import os

app = Flask(__name__)

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DETECTION_MODEL_PATH = os.path.join(BASE_DIR, 'Model', 'blood_clot_detection_model.h5')
SEGMENTATION_MODEL_DIR = os.path.join(BASE_DIR, 'Model', 'heart_segmentation_model.keras')
SEGMENTATION_CONFIG_PATH = os.path.join(SEGMENTATION_MODEL_DIR, 'config.json')
SEGMENTATION_WEIGHTS_PATH = os.path.join(SEGMENTATION_MODEL_DIR, 'model.weights.h5')
IMG_SIZE = 128


def load_segmentation_model():
    # Newer Keras can load the packaged directory directly, but fallback handles
    # environments where directory loading is treated as a file path.
    try:
        return tf.keras.models.load_model(SEGMENTATION_MODEL_DIR)
    except Exception:
        with open(SEGMENTATION_CONFIG_PATH, 'r', encoding='utf-8') as f:
            model_config = json.load(f)
        model = tf.keras.models.model_from_json(json.dumps(model_config))
        model.load_weights(SEGMENTATION_WEIGHTS_PATH)
        return model


print('Loading models...')
detection_model = tf.keras.models.load_model(DETECTION_MODEL_PATH)
segmentation_model = load_segmentation_model()
print('Models loaded successfully!')


def encode_image_to_data_url(image):
    success, buffer = cv2.imencode('.png', image)
    if not success:
        raise ValueError('Failed to encode image.')
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return f'data:image/png;base64,{image_base64}'


def run_segmentation(image_bgr):
    original_h, original_w = image_bgr.shape[:2]

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    gray_input = gray_resized.astype(np.float32) / 255.0
    gray_input = np.expand_dims(gray_input, axis=(0, -1))

    pred_mask = segmentation_model.predict(gray_input, verbose=0)[0, :, :, 0]
    binary_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    binary_mask = cv2.resize(binary_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    color_mask = np.zeros_like(image_bgr)
    color_mask[:, :, 2] = binary_mask
    overlay = cv2.addWeighted(image_bgr, 0.75, color_mask, 0.25, 0)

    return binary_mask, overlay


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Invalid image file'})

        det_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        det_input = np.expand_dims(det_img, axis=0)

        prediction = detection_model.predict(det_input, verbose=0)
        score = float(prediction[0][0])

        if score > 0.5:
            label = 'SICK (Blood Clot Detected)'
            confidence = f'{score * 100:.2f}%'
            status = 'danger'

            seg_mask, seg_overlay = run_segmentation(img)
            segmentation_enabled = True
            segmentation_mask = encode_image_to_data_url(seg_mask)
            segmentation_overlay = encode_image_to_data_url(seg_overlay)
        else:
            label = 'NORMAL (Healthy)'
            confidence = f'{(1 - score) * 100:.2f}%'
            status = 'success'

            segmentation_enabled = False
            segmentation_mask = None
            segmentation_overlay = None

        return jsonify({
            'label': label,
            'confidence': confidence,
            'status': status,
            'segmentation_enabled': segmentation_enabled,
            'segmentation_mask': segmentation_mask,
            'segmentation_overlay': segmentation_overlay
        })

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
