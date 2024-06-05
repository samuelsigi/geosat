import os
import cv2
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO

# Function to open file dialog and get the video file path
def select_video_file():
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
    if file_path:
        video_entry.delete(0, tk.END)
        video_entry.insert(0, file_path)

# Function to start object detection on the selected video file
def start_object_detection():
    video_path = video_entry.get()
    if video_path:
        # Load the pre-trained YOLO model
        model_path = os.path.join('.', 'kedy-50.pt')
        model = YOLO(model_path)

        # Threshold for object detection confidence
        threshold = 0.2

        # Class names dictionary
        class_name_dict = {0: 'smoke', 1: 'fire'}

        cap = cv2.VideoCapture(video_path)
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame to fit screen
            frame = cv2.resize(frame, (screen_width, screen_height))

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

            # Display the frame
            cv2.imshow('Object Detection', frame)

            # Check for 'q' key press to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()

# Create the main application window
root = tk.Tk()
root.title("Object Detection")

# Create UI components
video_label = tk.Label(root, text="Select Video File:")
video_label.pack()

video_entry = tk.Entry(root, width=50)
video_entry.pack()

browse_button = tk.Button(root, text="Browse", command=select_video_file)
browse_button.pack()

start_button = tk.Button(root, text="Start Detection", command=start_object_detection)
start_button.pack()

# Run the main event loop
root.mainloop()
