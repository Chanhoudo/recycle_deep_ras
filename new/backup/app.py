import base64
import io
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tflite_runtime.interpreter as tflite
from flask import Flask, render_template, jsonify
from picamera2 import Picamera2
import time

app = Flask(__name__)

# TFLite �� �ε�
interpreter = tflite.Interpreter(model_path="best.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Ŭ���� �� ���� (����)
labels = ["can", "nothing", "paper", "pet"]

# Picamera2 �ʱ�ȭ
picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration())
picam2.start()

def preprocess_image(frame):
    """���� �Է� ���Ŀ� �°� �̹����� ��ó���մϴ�."""
    image = cv2.resize(frame, (640, 640))  # ���� �Է� ũ�⿡ �°� ����
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)  # ��ġ ���� �߰�
    image = np.transpose(image, (0, 3, 1, 2))  # HWC���� CHW�� ����
    return image

def predict_image(frame):
    """�̹����� ���� �߷��� �����ϰ� ������ Ŭ������ ��ȯ�մϴ�."""
    input_data = preprocess_image(frame)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    class_probs = np.sum(output_data[0, :, 5:], axis=0)
    predicted_class = np.argmax(class_probs)
    return labels[predicted_class], class_probs[predicted_class], class_probs

def generate_label_image(frame):
    """�� ������ �������� �󺧸��� �̹����� �����մϴ�."""
    label, confidence, class_probs = predict_image(frame)
    
    # �ٿ�� �ڽ��� �׸��� �κ� (����� �󺧰� �ŷڵ��� �߰�)
    frame = cv2.putText(frame, f"Label: {label}, Confidence: {confidence:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    ret, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()

def generate_graph(class_probs):
    """�� ���� Ȯ���� �������� �׷����� �����մϴ�."""
    # Ŭ���� Ȯ���� 0~1�� ����ȭ
    class_probs = class_probs / np.max(class_probs)  # �ִ밪���� ������ 0~1�� ����ȭ

    plt.figure(figsize=(8, 4))  # ũ�� ����
    plt.bar(labels, class_probs, color='blue')
    plt.xlabel('Labels')
    plt.ylabel('Probability')
    plt.ylim(0, 1)  # Y�� ������ 0~1�� ����
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
    frame = picam2.capture_array()

    if frame is None:
        return jsonify(error="Failed to capture image"), 500

    label_image_data = generate_label_image(frame)
    _, _, class_probs = predict_image(frame)
    graph_img = generate_graph(class_probs)

    return jsonify(
        labelled_image_base64=base64.b64encode(label_image_data).decode('utf-8'),
        graph_img=graph_img,
        label=labels[np.argmax(class_probs)]
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
