from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import tensorflow as tf
import os
from werkzeug.utils import secure_filename
import base64

app = Flask(__name__)

# Load TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="emotion_detection_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to predict emotion on a face using TensorFlow Lite model
def predict_emotion(face_roi):
    # Resize and normalize the face region
    resized_face = cv2.resize(face_roi, (48, 48)) / 255.0

    # Reshape for model input
    input_data = np.expand_dims(resized_face, axis=0)
    input_data = np.expand_dims(input_data, axis=-1)

    # Ensure input data type is FLOAT32
    input_data = input_data.astype(np.float32)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Get the predicted emotion
    emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    predicted_emotion = emotion_labels[np.argmax(output_data)]

    return predicted_emotion

# Function to process video frames from camera
def generate_frames():
    # Open the camera
    cap = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert the frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4)

        # Iterate over detected faces
        for (x, y, w, h) in faces:
            # Extract the face region
            face_roi = gray_frame[y:y+h, x:x+w]

            # Predict emotion on the face using TensorFlow Lite model
            emotion = predict_emotion(face_roi)

            # Draw bounding box around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 51, 154), 2)

            # Display the predicted emotion with a background box
            text = emotion
            
            # Get text size
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]

            # Calculate text position
            text_x = x
            text_y = y - 10

            # Add padding to the text position
            text_padding = 5

            # Draw background rectangle
            background_x1 = x
            background_y1 = y - text_size[1] - 2*text_padding - 5
            background_x2 = x + text_size[0] + 2*text_padding
            background_y2 = y

            # Draw background rectangle
            cv2.rectangle(frame, (background_x1, background_y1), (background_x2, background_y2), (255, 51, 154), cv2.FILLED)

            # Draw text
            cv2.putText(frame, text, (text_x + text_padding, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame in the byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Release the camera
    cap.release()

# Function to predict emotion on uploaded image
def predict_emotion_image(image):
    # Convert the image to grayscale for face detection
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4)

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Extract the face region
        face_roi = gray_frame[y:y+h, x:x+w]

        # Predict emotion on the face using TensorFlow Lite model
        emotion = predict_emotion(face_roi)

        # Draw bounding box around the face
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 51, 154), 2)

        # Display the predicted emotion with a background box
        text = emotion

        # Get text size
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]  # Increased font scale

        # Calculate text position
        text_x = x
        text_y = y - 10

        # Add padding to the text position
        text_padding = 10  # Increased padding

        # Draw background rectangle
        background_x1 = x - text_padding
        background_y1 = y - text_size[1] - 2*text_padding
        background_x2 = x + text_size[0] + 2*text_padding
        background_y2 = y

        # Draw background rectangle
        cv2.rectangle(image, (background_x1, background_y1), (background_x2, background_y2), (255, 51, 154), cv2.FILLED)

        # Draw text
        cv2.putText(image, text, (text_x + text_padding, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)  # Increased font scale

    # Encode the processed image as JPEG
    _, buffer = cv2.imencode('.jpg', image)
    result_image = base64.b64encode(buffer).decode('utf-8')

    return result_image



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/real_time')
def real_time():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        result_image = predict_emotion_image(image)
        
        return render_template('index.html', result_image=result_image)

if __name__ == '__main__':
    app.run(debug=True)
