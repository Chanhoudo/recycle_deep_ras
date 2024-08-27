from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image

app = Flask(__name__)

# Initialize the camera
camera = cv2.VideoCapture(0)

# Load the TFLite model
interpreter = tflite.Interpreter(model_path="best.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define labels for the model's classes
labels = ["can", "nothing", "paper", "pet"]

def preprocess_image(frame):
    """Preprocess the image to match the input shape of the model."""
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = image.resize((input_details[0]['shape'][2], input_details[0]['shape'][1]))
    image = np.array(image).astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.transpose(image, (0, 3, 1, 2))  # Change from HWC to CHW
    return image

def predict_image(frame):
    """Run inference on the image and return the predicted class."""
    input_data = preprocess_image(frame)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    if output_data.ndim == 3:  # Assume shape (1, num_boxes, num_classes+5)
        class_probs = np.sum(output_data[0, :, 5:], axis=0)
        predicted_class = np.argmax(class_probs)
        return labels[predicted_class], class_probs[predicted_class]
    else:
        return "Unknown", 0

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Predict the label of the current frame
            label, confidence = predict_image(frame)
            # Display the label on the frame
            cv2.putText(frame, f"Label: {label}, Confidence: {confidence:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

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
    success, frame = camera.read()
    if success:
        label, confidence = predict_image(frame)
        return jsonify(label=label, confidence=confidence)
    else:
        return jsonify(label="None", confidence=0)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
