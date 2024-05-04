import cv2
import numpy as np

# Load the pre-trained model and its configuration
model_path = 'mobilenet_iter_73000.caffemodel'
config_path = 'deploy.prototxt'

# Load the DNN model from Caffe files
net = cv2.dnn.readNetFromCaffe(config_path, model_path)

# Classes the model can recognize (COCO dataset classes)
class_names = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
    'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
    'sofa', 'train', 'tvmonitor'
]

# Set up webcam capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Create a blob from the frame and set the input for the network
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)

    # Perform object detection
    detections = net.forward()

    # Loop over detections and draw bounding boxes for high confidence objects
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Only consider detections with high confidence
            class_id = int(detections[0, 0, i, 1])
            class_name = class_names[class_id]

            # Get bounding box coordinates
            box = detections[0, 0, i, 3:7] * np.array(
                [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (x1, y1, x2, y2) = box.astype("int")

            # Draw the bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name}: {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow('Object Recognition', frame)

    # Break loop on 'Enter' key
    if cv2.waitKey(1) == 13:
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
