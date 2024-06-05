import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, Response, jsonify
import cv2
from ultralytics import YOLO
from flask import Flask, render_template, request, Response
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import subprocess
import random
import tkinter as tk
from tkinter import ttk, filedialog
from io import BytesIO
import base64
from shapely.geometry import Point
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.patches import Patch
from matplotlib.figure import Figure
import matplotlib.ticker as ticker
import tempfile
import os
import io
import base64
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

app = Flask(__name__)

# Function to start object detection on the selected video file
def start_object_detection(video_path):
    # Load the pre-trained YOLO model
    model_path = './kedy-50.pt'
    model = YOLO(model_path)

    # Threshold for object detection confidence
    threshold = 0.2

    # Class names dictionary
    class_name_dict = {0: 'smoke', 1: 'fire'}

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection on the frame
        results = model(frame)[0]

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > threshold:
                # Draw bounding box around the detected object
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                # Put text label indicating the class of the object
                cv2.putText(frame, class_name_dict[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        # Convert frame to JPEG format
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()

        # Yield frame as a byte string for streaming response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    # Release resources
    cap.release()

# Function to create and display the bar graph in a Tkinter window
def show_bar_graph_tkinter():
    # Define the 21 predefined values
    values = [33, 31, 60, 48, 47.5, 53, 63, 54, 36, 51, 58, 47, 70, 68, 100, 94, 60, 53, 65, 71, 68]
    years = list(range(2002, 2023))
    year_labels = [str(year) for year in years]  # Convert years to string labels

    # Create a Tkinter window
    window = tk.Tk()
    window.title("Bar Graph")

    # Create a matplotlib figure
    fig = Figure(figsize=(10, 6))  # Adjust the figure size as needed

    # Add the bar graph to the figure
    ax = fig.add_subplot(111)
    ax.bar(years, values, color='yellowgreen', width=0.8)  # Adjust the width of the bars
    ax.set_xlabel('Years')
    ax.set_ylabel('Change Rate')
    ax.set_title('Forest Cover Change from 2002 to 2022')

    # Set the x-axis tick labels
    ax.set_xticks(years)  # Set the tick positions
    ax.set_xticklabels(year_labels, rotation=45, ha='right', fontsize=8)  # Set the tick labels

    # Convert the matplotlib figure to a Tkinter canvas
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack()

    # Run the Tkinter event loop
    window.mainloop()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/fire_detection_page')
def fire_detection_page():
    return render_template('fire_detection_page.html')

@app.route('/visualization_page')
def visualization_page():
    return render_template('visualization_page.html')

@app.route('/year_wise_page')
def run_tkinter_script():
    # Execute the Tkinter script using subprocess
    subprocess.Popen(['python', 'year_wise.py'])
    # Optionally, you can return a response to indicate that the script is being executed
    return ''

@app.route('/compare_page', methods=['POST'])
def run_tkinter():
    # Code to run the tkinter script here
    subprocess.Popen(['python', 'compare.py'])
    return ''

# Route to display the bar graph in a Tkinter window
@app.route('/show_bar_graph_tkinter', methods=['GET'])
def show_bar_graph_tkinter_route():
    show_bar_graph_tkinter()
    return ''


@app.route('/fire_detection', methods=['POST'])
def fire_detection():
    if 'file' not in request.files:
        return 'No file part'

    video_file = request.files['file']
    if video_file.filename == '':
        return 'No selected file'

    # Save the uploaded video file temporarily
    video_path = 'uploaded_video.mp4'
    video_file.save(video_path)

    # Return the streaming response for object detection
    return Response(start_object_detection(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)