import numpy as np
import cv2
import tflite_runtime.interpreter as tflite

# 모델 경로 설정
MODEL_PATH = "best.tflite"  # TFLite 모델 파일 경로
IMAGE_PATH = "static/original/image1.jpg"  # 테스트할 이미지 파일 경로

# TensorFlow Lite 모델 로드 및 초기화
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 클래스 라벨 (예제)
labels = ["can", "nothing", "paper", "pet"]

def preprocess_image(image_path, input_size):
    """이미지를 전처리하여 모델 입력에 맞게 변환합니다."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")
    
    original_image = image.copy()  # 나중에 바운딩 박스를 그리기 위해 원본 이미지 유지
    image = cv2.resize(image, input_size)
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)  # 배치 차원 추가
    return original_image, image

def detect_objects(interpreter, input_image):
    """이미지에서 객체를 탐지하고 바운딩 박스 및 클래스 정보를 반환합니다."""
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()
    
    # YOLO 모델의 경우, 각 앵커 박스마다 바운딩 박스, 클래스 및 신뢰도 값이 출력됩니다.
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    boxes = output_data[:, :4]  # 바운딩 박스 좌표 (ymin, xmin, ymax, xmax)
    confidences = output_data[:, 4]  # 바운딩 박스 신뢰도 (IOU)
    class_probs = output_data[:, 5:]  # 클래스 확률
    return boxes, confidences, class_probs

def filter_boxes(boxes, confidences, class_probs, threshold=0.4):
    """신뢰도가 일정 임계값 이상인 바운딩 박스만 필터링합니다."""
    class_ids = np.argmax(class_probs, axis=-1)
    max_probs = np.max(class_probs, axis=-1)
    mask = max_probs * confidences > threshold  # 신뢰도와 클래스 확률이 모두 높아야 함
    
    filtered_boxes = boxes[mask]
    filtered_confidences = confidences[mask]
    filtered_class_ids = class_ids[mask]
    return filtered_boxes, filtered_confidences, filtered_class_ids

def draw_bounding_boxes(image, boxes, confidences, class_ids):
    """이미지에 바운딩 박스를 그립니다."""
    h, w, _ = image.shape
    for box, confidence, class_id in zip(boxes, confidences, class_ids):
        ymin, xmin, ymax, xmax = box
        start_point = (int(xmin * w), int(ymin * h))
        end_point = (int(xmax * w), int(ymax * h))
        color = (255, 0, 0)
        image = cv2.rectangle(image, start_point, end_point, color, 2)
        label = f"{labels[class_id]}: {confidence:.2f}"
        image = cv2.putText(image, label, (start_point[0], start_point[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

def main(image_path):
    """메인 함수: 이미지에서 객체 탐지 후 바운딩 박스를 그려서 표시합니다."""
    input_size = (input_details[0]['shape'][2], input_details[0]['shape'][3])  # 입력 이미지 크기
    original_image, input_image = preprocess_image(image_path, input_size)
    
    boxes, confidences, class_probs = detect_objects(interpreter, input_image)
    filtered_boxes, filtered_confidences, filtered_class_ids = filter_boxes(boxes, confidences, class_probs)
    
    image_with_boxes = draw_bounding_boxes(original_image, filtered_boxes, filtered_confidences, filtered_class_ids)
    
    # 결과 출력
    cv2.imshow("Detected Objects", image_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(IMAGE_PATH)
