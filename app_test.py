import base64
import io
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tflite_runtime.interpreter as tflite
from flask import Flask, render_template, jsonify, request, send_file
import subprocess
import datetime

app = Flask(__name__)

# 디렉토리 설정
ORIGINAL_IMAGE_DIRECTORY = '/home/aisw/yolov5/data/images'
LABEL_IMAGE_DIRECTORY = '/home/aisw/Downloads/label_image'
LABEL_INFO_DIRECTORY = '/home/aisw/Downloads/label_info'

def create_interpreter():
    """새로운 TFLite 인터프리터 객체를 생성합니다."""
    interpreter = tflite.Interpreter(model_path="best.tflite")
    interpreter.allocate_tensors()
    return interpreter

# 클래스 라벨 정의
labels = ["can", "nothing", "paper", "pet"]

def capture_image():
    """이미지를 캡처하고 저장합니다."""
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    image_path = f'{ORIGINAL_IMAGE_DIRECTORY}/image_{timestamp}.jpg'

    if not os.path.exists(ORIGINAL_IMAGE_DIRECTORY):
        os.makedirs(ORIGINAL_IMAGE_DIRECTORY)

    try:
        # 미리보기 없이 사진 촬영
        subprocess.run(['libcamera-still', '-o', image_path, '--nopreview', '--timeout', '1000'], check=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        return None
    return image_path

def preprocess_image(image_path):
    """모델의 입력 형식에 맞게 이미지를 전처리합니다."""
    frame = cv2.imread(image_path)
    if frame is None:
        return None

    image = cv2.resize(frame, (640, 640))
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.transpose(image, (0, 3, 1, 2))
    return image

def predict_image(image_path):
    """이미지에 대해 추론을 실행하고 예측된 클래스와 바운딩 박스 좌표를 반환합니다."""
    input_data = preprocess_image(image_path)
    if input_data is None:
        return "None", 0.0, np.zeros(len(labels)), None

    interpreter = create_interpreter()
    # 모델의 입력 텐서 설정
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)

    # 예측 수행
    interpreter.invoke()

    # 모델의 출력 텐서 가져오기
    output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index']).copy()

    # 바운딩 박스와 클래스 확률 처리
    boxes = output_data[0, :, :4]
    class_probs = np.sum(output_data[0, :, 5:], axis=0)
    predicted_class = np.argmax(class_probs)

    return labels[predicted_class], class_probs[predicted_class], class_probs, boxes

def generate_label_image(original_image_path):
    """모델 예측을 바탕으로 라벨링된 이미지를 생성하고, 라벨 정보를 파일로 저장합니다."""
    frame = cv2.imread(original_image_path)
    if frame is None:
        print("Failed to read the image.")
        return None

    label, confidence, class_probs, boxes = predict_image(original_image_path)

    # 텍스트 파일로 라벨 정보 저장
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    label_info_filename = f'{LABEL_INFO_DIRECTORY}/label_info.txt'

    if not os.path.exists(LABEL_INFO_DIRECTORY):
        os.makedirs(LABEL_INFO_DIRECTORY)

    with open(label_info_filename, 'a') as f:  # 'a' 모드로 파일 열기: 파일이 존재하면 끝에 추가
        f.write(f"{timestamp}_{label}\n")  # 시간과 라벨 정보만 기록

    if boxes is not None:
        h, w, _ = frame.shape
        for box in boxes:
            x, y, box_w, box_h = box
            x1, y1 = int(x * w), int(y * h)
            x2, y2 = int((x + box_w) * w), int((y + box_h) * h)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Label: {label}, Confidence: {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3, cv2.LINE_AA)  # 큰 글자 크기와 두께

    result_text = f"result: {label}, Confidence: {confidence:.2f}"
    cv2.putText(frame, result_text, (10, frame.shape[0] - 20),  # 위치를 하단으로 이동
                cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 5, cv2.LINE_AA)  # 글자 크기 및 두께 조정

    output_path = os.path.join(LABEL_IMAGE_DIRECTORY, 'labelled_image.jpg')
    if not os.path.exists(LABEL_IMAGE_DIRECTORY):
        os.makedirs(LABEL_IMAGE_DIRECTORY)

    # 이미지를 파일로 저장
    cv2.imwrite(output_path, frame)

    # 라벨링된 이미지를 Base64로 인코딩하여 반환
    ret, buffer = cv2.imencode('.jpg', frame)
    if ret:
        return buffer.tobytes()
    else:
        print("Failed to encode the image.")
        return None

def generate_graph(image_path):
    """모델 예측 확률을 바탕으로 그래프를 생성합니다."""
    _, _, class_probs, _ = predict_image(image_path)

    class_probs = class_probs / np.max(class_probs)

    plt.figure(figsize=(8, 4))
    plt.bar(labels, class_probs, color='blue')
    plt.xlabel('Labels')
    plt.ylabel('Probability')
    plt.ylim(0, 1)
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
    
    # 이미지 촬영
    image_path = capture_image()
    if image_path is None:
        return jsonify(error="Failed to capture image"), 500

    label_image_data = generate_label_image(image_path)
    if label_image_data is None:
        return jsonify(error="Failed to generate label image"), 500

    graph_img = generate_graph(image_path)

    # 예측 클래스의 확률 배열을 얻어 라벨을 설정
    _, _, class_probs, _ = predict_image(image_path)
    predicted_label = labels[np.argmax(class_probs)]

    # 현재 시간 가져오기
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    detection_info = f"{timestamp}_{predicted_label}"

    # 기존 반환 데이터에 detection_info 추가
    return jsonify(
        labelled_image_base64=base64.b64encode(label_image_data).decode('utf-8'),
        graph_img=graph_img,
        label=predicted_label,
        detection_info=detection_info  # 시간_라벨값 추가
    )

@app.route('/save_history', methods=['POST'])
def save_history():
    try:
        # request 모듈에서 JSON 데이터 추출
        history_data = request.json.get('history', '')

        # 현재 날짜를 가져와 파일 이름을 생성합니다.
        today_date = datetime.datetime.now().strftime('%Y%m%d')
        file_name = f"{today_date}_data.txt"
        file_path = os.path.join(LABEL_INFO_DIRECTORY, file_name)

        # 파일에 검지 이력 저장
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(history_data)

        # 파일을 클라이언트에 전송
        return send_file(file_path, as_attachment=True)

    except Exception as e:
        # 오류 발생 시 로그에 기록하고 클라이언트에 오류 메시지 전송
        print(f"Error: {e}")
        return jsonify({"error": "파일 저장 중 오류가 발생했습니다."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
