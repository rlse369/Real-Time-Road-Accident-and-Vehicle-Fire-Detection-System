import os
import tempfile
from datetime import datetime

import ultralytics
from flask import Flask, request, render_template, Response, redirect, url_for
import cv2
import math
import numpy as np

app = Flask(__name__)
app.secret_key = '123456'

UPLOAD_FOLDER = 'samples'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
RESULT_FOLDER = 'results'
app.config['RESULT_FOLDER'] = RESULT_FOLDER

font = cv2.FONT_HERSHEY_SIMPLEX

model = ultralytics.YOLO('fire.pt')
from detection import AccidentDetectionModel
model1 = AccidentDetectionModel("model.json", 'model_weights.h5')


@app.route('/')
def index():
    return render_template('index.html')


def video_stream():
    video = 'a&f_demo.mov'
    cap = cv2.VideoCapture(video)

    frame_count = 0
    frames_folder = 'frames/demo'
    os.makedirs(frames_folder,  exist_ok=True)

    # Read until video is completed
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))

        # Fire Detection
        classnames = ['Fire']

        # Accident Detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(gray_frame, (250, 250))
        pred, prob = model1.predict_accident(roi[np.newaxis, :, :])

        # Check for Accident
        if pred == "Accident":
            prob = round(prob[0][0] * 100, 2)
            if prob > 80:
                cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
                cv2.putText(frame, pred + " " + str(prob) + "%", (20, 30), font, 1, (255, 255, 0), 2)
            else:
                cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
                cv2.putText(frame, "No Accident", (20, 30), font, 1, (255, 255, 0), 2)

        # Check for Fire
        results = model.predict(source=frame, imgsz=640, conf=0.6)
        for info in results:
            boxes = info.boxes
            for box in boxes:
                confidence = box.conf[0]
                confidence = math.ceil(confidence * 100)
                Class = int(box.cls[0])
                if confidence > 60:
                    cv2.rectangle(frame, (0, 40), (280, 80), (0, 0, 0), -1)
                    cv2.putText(frame, f'{classnames[Class]} {confidence}%', (20, 70), font, 1, (255, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (0, 40), (280, 40), (0, 0, 0), -1)
                    cv2.putText(frame, "No Fire", (20, 30), font, 1, (255, 255, 0), 2)

        frame_path = os.path.join(frames_folder,f'frame_{frame_count}.jpg')
        cv2.imwrite(frame_path,frame)
        frame_count += 1

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()

        yield (b'--frame\r\n' 
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# Define route for real-time detection page
@app.route('/real_time_detection')
def real_time_detection():
    return Response(video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Define video feed route
@app.route('/video_detection')
def video_feed():
    return render_template('video_detection.html')


@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        files = request.files.getlist('files')

        # Create a temporary directory to store the uploaded files
        with (tempfile.TemporaryDirectory() as temp_dir):
            # Save each uploaded file to the temporary directory
            file_paths = []
            for file in files:
                file_name, file_extension = os.path.splitext(file.filename)
                file_path = os.path.join(temp_dir, file.filename)
                file.save(file_path)
                file_paths.append(file_path)

            # Process each uploaded file
            result_images = []
            for file_path in file_paths:
                img = cv2.imread(file_path)
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                roi = cv2.resize(gray_img, (250, 250))

                pred, prob = model1.predict_accident(roi[np.newaxis, :, :])

                if pred == "Accident":
                    prob = round(prob[0][0] * 100, 2)
                    if prob > 90:
                        # Save the snapshot with prediction
                        snapshot_path = f"static/snapshots/{file_name}_accident.jpg"
                        cv2.imwrite(snapshot_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                        result_text = pred + " " + str(prob) + "%"
                        result_color = (0, 0, 255)  # red

                    else:
                        result_text = "No Accident"
                        result_color = (0, 255, 0)  # green

                # Perform inference
                results = model.predict(source=img, imgsz=640, conf=0.6)
                for info in results:
                    boxes = info.boxes
                    for box in boxes:
                        confidence = box.conf[0]
                        confidence = math.ceil(confidence * 100)
                        if confidence > 60:
                            # Save the snapshot with prediction
                            snapshot_path = f"static/snapshots/{file_name}_fire.jpg"
                            cv2.imwrite(snapshot_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                            result_text2 = f"Fire" + " " + str(confidence) + "%"
                            result_color2 = (0, 0, 255)  # red
                        else:
                            result_text2 = "No Fire"
                            result_color2 = (0, 255, 0)  # green

                # Draw result on image
                cv2.putText(img, result_text, (20, 30), font, 1, result_color, 2)
                cv2.putText(img, result_text2, (20, 60), font, 1, result_color2, 2)

                # Create a new folder for each result image
                result_folder_name = os.path.splitext(os.path.basename(file_path))[0]
                dt = datetime.now().strftime("%Y%m%d_%H%M")
                result_folder = os.path.join(f"results/{dt}", result_folder_name)
                os.makedirs(result_folder, exist_ok=True)

                # Save the result image in the new folder
                result_image_path = os.path.join(result_folder, "result_image.jpg")
                cv2.imwrite(result_image_path, img)

                # Append the result image path to the list
                result_images.append(result_image_path)
                print(result_images)

            return render_template('base.html', result_images=result_images)

    return render_template('upload.html')


if __name__ == "__main__":
    app.run(debug=True)
