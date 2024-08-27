from flask import Flask, render_template, jsonify, url_for
import os
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# ���丮 ����
ORIGINAL_IMAGE_DIRECTORY = 'static/original'
ORIGINAL_IMAGE_FILES = sorted([f for f in os.listdir(ORIGINAL_IMAGE_DIRECTORY) if f.endswith(('.jpg', '.jpeg'))])
current_image_index = 0

# TFLite �� �ε�
interpreter = tflite.Interpreter(model_path="best.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Ŭ���� �� ���� (����)
labels = ["can", "nothing", "paper", "pet"]

def preprocess_image(image_path):
    """���� �Է� ���Ŀ� �°� �̹����� ��ó���մϴ�."""
    frame = cv2.imread(image_path)
    if frame is None:
        return None

    image = cv2.resize(frame, (640, 640))  # ���� �Է� ũ�⿡ �°� ����
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)  # ��ġ ���� �߰�
    image = np.transpose(image, (0, 3, 1, 2))  # HWC���� CHW�� ����
    return image

def predict_image(image_path):
    """�̹����� ���� �߷��� �����ϰ� ������ Ŭ������ ��ȯ�մϴ�."""
    input_data = preprocess_image(image_path)
    if input_data is None:
        return "None", 0.0, np.zeros(len(labels))

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    class_probs = np.sum(output_data[0, :, 5:], axis=0)
    predicted_class = np.argmax(class_probs)
    return labels[predicted_class], class_probs[predicted_class], class_probs

def generate_label_image(original_image_path):
    """�� ������ �������� �󺧸��� �̹����� �����մϴ�."""
    frame = cv2.imread(original_image_path)
    if frame is None:
        return None

    label, confidence, _ = predict_image(original_image_path)
    frame = cv2.putText(frame, f"Label: {label}, Confidence: {confidence:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    ret, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()

def generate_graph():
    """���� �׷����� �����մϴ�."""
    x = labels
    y = np.random.rand(len(labels))  # ���⿡ ���� Ŭ���� Ȯ���� ����� �� �ֽ��ϴ�

    plt.figure(figsize=(8, 4))  # ũ�� ����
    plt.bar(x, y, color='blue')
    plt.xlabel('Labels')
    plt.ylabel('Probability')
    plt.title('Class Probabilities')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    graph_img = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return graph_img

@app.route('/')
def index():
    """Ȩ �������� �������մϴ�."""
    return render_template('index.html')

@app.route('/image_data')
def image_data():
    """���� �̹����� �׷����� ��ȯ�մϴ�."""
    global current_image_index
    image_file = ORIGINAL_IMAGE_FILES[current_image_index]
    image_path = os.path.join(ORIGINAL_IMAGE_DIRECTORY, image_file)

    original_image_url = url_for('static', filename=f'original/{image_file}')
    label_image_data = generate_label_image(image_path)
    if label_image_data is None:
        return jsonify(error="Failed to generate label image"), 500

    graph_img = generate_graph()

    # �̹��� �ε����� �������� �̵��ϰ�, ������ �̹����� ��� ù ��° �̹����� ���ư�
    current_image_index = (current_image_index + 1) % len(ORIGINAL_IMAGE_FILES)

    return jsonify(
        original_image_url=original_image_url,
        labelled_image_base64=base64.b64encode(label_image_data).decode('utf-8'),
        graph_img=graph_img
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
