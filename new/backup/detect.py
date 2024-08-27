import numpy as np
import cv2
import tflite_runtime.interpreter as tflite

# �� ��� ����
MODEL_PATH = "best.tflite"  # TFLite �� ���� ���
IMAGE_PATH = "static/original/image1.jpg"  # �׽�Ʈ�� �̹��� ���� ���

# TensorFlow Lite �� �ε� �� �ʱ�ȭ
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Ŭ���� �� (����)
labels = ["can", "nothing", "paper", "pet"]

def preprocess_image(image_path, input_size):
    """�̹����� ��ó���Ͽ� �� �Է¿� �°� ��ȯ�մϴ�."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"�̹����� �ҷ��� �� �����ϴ�: {image_path}")
    
    original_image = image.copy()  # ���߿� �ٿ�� �ڽ��� �׸��� ���� ���� �̹��� ����
    image = cv2.resize(image, input_size)
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)  # ��ġ ���� �߰�
    return original_image, image

def detect_objects(interpreter, input_image):
    """�̹������� ��ü�� Ž���ϰ� �ٿ�� �ڽ� �� Ŭ���� ������ ��ȯ�մϴ�."""
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()
    
    # YOLO ���� ���, �� ��Ŀ �ڽ����� �ٿ�� �ڽ�, Ŭ���� �� �ŷڵ� ���� ��µ˴ϴ�.
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    boxes = output_data[:, :4]  # �ٿ�� �ڽ� ��ǥ (ymin, xmin, ymax, xmax)
    confidences = output_data[:, 4]  # �ٿ�� �ڽ� �ŷڵ� (IOU)
    class_probs = output_data[:, 5:]  # Ŭ���� Ȯ��
    return boxes, confidences, class_probs

def filter_boxes(boxes, confidences, class_probs, threshold=0.4):
    """�ŷڵ��� ���� �Ӱ谪 �̻��� �ٿ�� �ڽ��� ���͸��մϴ�."""
    class_ids = np.argmax(class_probs, axis=-1)
    max_probs = np.max(class_probs, axis=-1)
    mask = max_probs * confidences > threshold  # �ŷڵ��� Ŭ���� Ȯ���� ��� ���ƾ� ��
    
    filtered_boxes = boxes[mask]
    filtered_confidences = confidences[mask]
    filtered_class_ids = class_ids[mask]
    return filtered_boxes, filtered_confidences, filtered_class_ids

def draw_bounding_boxes(image, boxes, confidences, class_ids):
    """�̹����� �ٿ�� �ڽ��� �׸��ϴ�."""
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
    """���� �Լ�: �̹������� ��ü Ž�� �� �ٿ�� �ڽ��� �׷��� ǥ���մϴ�."""
    input_size = (input_details[0]['shape'][2], input_details[0]['shape'][3])  # �Է� �̹��� ũ��
    original_image, input_image = preprocess_image(image_path, input_size)
    
    boxes, confidences, class_probs = detect_objects(interpreter, input_image)
    filtered_boxes, filtered_confidences, filtered_class_ids = filter_boxes(boxes, confidences, class_probs)
    
    image_with_boxes = draw_bounding_boxes(original_image, filtered_boxes, filtered_confidences, filtered_class_ids)
    
    # ��� ���
    cv2.imshow("Detected Objects", image_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(IMAGE_PATH)
