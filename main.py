import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image

# Load YOLOv5 model with custom weights file
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')


# Define function to perform object detection on input image and return output image with bounding boxes and labels
def detect_objects(image):
    # Convert PIL image to OpenCV format
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Perform object detection using YOLOv5 model
    results = model(image)

    # Draw bounding boxes and labels on output image
    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, class_id = result.tolist()
        label = model.names[int(class_id)]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, f'{label} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0),
                    2)

    # Convert output image to PIL format
    output_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    return output_image


# Define Streamlit app
def app():
    # Set app title and page icon
    st.set_page_config(page_title='Object Detection', page_icon=':detective:')

    # Set app header
    st.header('Object Detection App')

    # Define file uploader
    file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

    # Check if file is uploaded
    if file is not None:
        # Display input image
        input_image = Image.open(file)
        st.image(input_image, caption='Input Image', use_column_width=True)

        # Perform object detection and display output image
        output_image = detect_objects(input_image)
        st.image(output_image, caption='Output Image', use_column_width=True)


# Run Streamlit app
if __name__ == '__main__':
    app()
