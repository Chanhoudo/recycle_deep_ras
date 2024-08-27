from flask import Flask, render_template, Response, jsonify, url_for
import os
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image
import matplotlib.pyplot as plt
import io
import base64
import cv2

app = Flask(__name__)

# 디렉토리 설정
IMAGE_DIRECTORY = '/home/aisw/static'
IMAGE_FILES = sorted([f for f in os.listdir(IMAGE_DIRECTORY) if f.endswith(('.jpg', '.jpeg', '.png'))])
current_image_index = 0

# TFLite 모델 로드
interpreter = tflite.Interpreter(model_path="best.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 클래스 라벨 정의
labels = ["can", "nothing", "paper", "pet"]

def preprocess_image(frame):
    """모델의 입력 형식에 맞게 이미지를 전처리합니다."""
    image = Image.fromarray(frame)
    image = image.resize((640, 640))  # 모델의 입력 크기에 맞게 조정
    image = np.array(image).astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)  # 배치 차원 추가
    image = np.transpose(image, (0, 3, 1, 2))  # HWC에서 CHW로 변경
    return image

def predict_image(frame):
    """이미지에 대해 추론을 실행하고 예측된 클래스를 반환합니다."""
    input_data = preprocess_image(frame)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    class_probs = np.sum(output_data[0, :, 5:], axis=0)
    predicted_class = np.argmax(class_probs)
    return labels[predicted_class], class_probs[predicted_class], class_probs

def generate_frames():
    global current_image_index
    while True:
        image_file = IMAGE_FILES[current_image_index]
        image_path = os.path.join(IMAGE_DIRECTORY, image_file)
        frame = cv2.imread(image_path)

        if frame is None:
            continue

        label, confidence, class_probs = predict_image(frame)
        frame = cv2.putText(frame, f"Label: {label}, Confidence: {confidence:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Update image index
        current_image_index = (current_image_index + 1) % len(IMAGE_FILES)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict')
def predict():
    image_file = IMAGE_FILES[current_image_index]
    image_path = os.path.join(IMAGE_DIRECTORY, image_file)
    frame = cv2.imread(image_path)

    if frame is None:
        return jsonify(label="None", confidence=0, image_url=url_for('static', filename=image_file), graph_img='')

    label, confidence, class_probs = predict_image(frame)

    # Create a bar plot of class probabilities
    plt.figure(figsize=(5, 3))
    plt.bar(labels, class_probs, color='blue')
    plt.xlabel('Labels')
    plt.ylabel('Probability')
    plt.title('Class Probabilities')
    plt.tight_layout()

    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    graph_img = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    return jsonify(
        label=label,
        confidence=confidence,
        image_url=url_for('static', filename=image_file),
        graph_img=graph_img
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
