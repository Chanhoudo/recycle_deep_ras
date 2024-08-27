from flask import Flask, render_template, jsonify, url_for
import os
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# 디렉토리 설정
ORIGINAL_IMAGE_DIRECTORY = 'static/original'
ORIGINAL_IMAGE_FILES = sorted([f for f in os.listdir(ORIGINAL_IMAGE_DIRECTORY) if f.endswith(('.jpg', '.jpeg'))])
current_image_index = 0

# TFLite 모델 로드
interpreter = tflite.Interpreter(model_path="best.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 클래스 라벨 정의 (예제)
labels = ["can", "nothing", "paper", "pet"]

def preprocess_image(image_path):
    """모델의 입력 형식에 맞게 이미지를 전처리합니다."""
    frame = cv2.imread(image_path)
    if frame is None:
        return None

    image = cv2.resize(frame, (640, 640))  # 모델의 입력 크기에 맞게 조정
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)  # 배치 차원 추가
    image = np.transpose(image, (0, 3, 1, 2))  # HWC에서 CHW로 변경
    return image

def predict_image(image_path):
    """이미지에 대해 추론을 실행하고 예측된 클래스를 반환합니다."""
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
    """모델 예측을 바탕으로 라벨링된 이미지를 생성합니다."""
    frame = cv2.imread(original_image_path)
    if frame is None:
        return None

    label, confidence, _ = predict_image(original_image_path)
    frame = cv2.putText(frame, f"Label: {label}, Confidence: {confidence:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    ret, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()

def generate_graph():
    """예시 그래프를 생성합니다."""
    x = labels
    y = np.random.rand(len(labels))  # 여기에 실제 클래스 확률을 사용할 수 있습니다

    plt.figure(figsize=(8, 4))  # 크기 조정
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
    """홈 페이지를 렌더링합니다."""
    return render_template('index.html')

@app.route('/image_data')
def image_data():
    """현재 이미지와 그래프를 반환합니다."""
    global current_image_index
    image_file = ORIGINAL_IMAGE_FILES[current_image_index]
    image_path = os.path.join(ORIGINAL_IMAGE_DIRECTORY, image_file)

    original_image_url = url_for('static', filename=f'original/{image_file}')
    label_image_data = generate_label_image(image_path)
    if label_image_data is None:
        return jsonify(error="Failed to generate label image"), 500

    graph_img = generate_graph()

    # 이미지 인덱스를 다음으로 이동하고, 마지막 이미지인 경우 첫 번째 이미지로 돌아감
    current_image_index = (current_image_index + 1) % len(ORIGINAL_IMAGE_FILES)

    return jsonify(
        original_image_url=original_image_url,
        labelled_image_base64=base64.b64encode(label_image_data).decode('utf-8'),
        graph_img=graph_img
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
