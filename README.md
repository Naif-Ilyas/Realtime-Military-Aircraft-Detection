Objective
• The undertaken project tackles realistic videos, including CCTV videos and real-time
images and videos. The videographer or the stationary camera will face the sky to get realtime images /video frames for processing and detecting the fighter and bomber jets.
• To develop a sky-based object tracking mechanism and thus the system can also be
deployed with real-time satellite images/video streaming.
• To design an algorithm that can extract the feature of the object that’s coming towards the
capturing unit and classifies the object into the type of aircraft.
Current Progress
Selected a dataset from Kaggle containing images and their labels. The algorithm used is YOLO (You
Only Look Once) for one-stage real-time object detection in videos and images. YOLOv5 is inspired by
YOLOv5 by Ultralytics and is developed by researchers at Meituan. YOLOv5 could be trained easily for
specific domains by providing the annotated data as a feed.
About The Dataset :
Size: 8 GB
Training Data : 5560 images & labels
Validation Data : 2781 images & labels
Aircraft Classes : 40
YOLOv5 by Ultralytics
Fast, precise and easy to train, YOLOv5 has a long and successful history of real time object
detection. Treat YOLOv5 as a university where you'll feed your model information for it to
learn from and grow into one integrated tool. The YOLO network consists of three main
pieces.
Backbone: A convolutional neural network that aggregates and forms image features at
different granularities.
Neck: A series of layers to mix and combine image features to pass them forward to
prediction.
Head: Consumes features from the neck and takes box and class prediction steps.
