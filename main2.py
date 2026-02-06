import cv2
import numpy as np
import streamlit as st
from PIL import Image
import torch

# Load YOLOv5 model with custom weights
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')


# Function to detect objects in an image and draw bounding boxes and labels
def detect_objects(image):
    # Convert PIL image to OpenCV format
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Perform object detection using YOLOv5 model
    results = model(image)

    # Draw bounding boxes and labels on output image
    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, class_id = result.tolist()
        label = model.names[int(class_id)]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), thickness=3)
        cv2.putText(image, f'{label} {conf:.2f}', (int(x1), int(y1) - 15), cv2.FONT_HERSHEY_COMPLEX, fontScale=1,
                    color=(0, 255, 0), thickness=2)

    # Convert output image to PIL format
    output_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    return output_image


# Streamlit UI for video detection
def video_detection():
    st.title("Object Detection in Video Stream")

    # Open video capture device (0 for camera, or path to video file)
    cap = cv2.VideoCapture('videoplayback_2_Trim.mp4')

    # Create a Streamlit placeholder for displaying the output image
    output_placeholder = st.empty()

    # Loop through frames and detect objects
    while True:
        # Read frame from video stream
        ret, frame = cap.read()
        if not ret:
            break

        # Convert OpenCV frame to PIL format
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Detect objects and get output image
        output_image = detect_objects(image)

        # Display output image with bounding boxes and labels
        output_placeholder.image(output_image, channels='RGB')

        # Check for key press to exit
        if cv2.waitKey(1) == ord('q'):
            break

    # Release video capture device and close all windows
    cap.release()
    cv2.destroyAllWindows()


# Run video detection function
if __name__ == '__main__':
    video_detection()
