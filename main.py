import os
import cv2
from ultralytics import YOLO
import supervision as sv

# Load a pretrained model
model = YOLO("/home/sail/pjt/runs/detect/train3/weights/last.pt")

bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

cap = cv2.VideoCapture(0)  # Input video stream

if not cap.isOpened():
    print("Unable to read camera feed")
    exit()

output_dir = 'output_feed'
os.makedirs(output_dir, exist_ok=True)

img_counter = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    # Run the model on the current frame
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    # Filter detections with confidence score above 0.80
    high_conf_detections = detections[detections.confidence > 0.65]

    # Annotate the image with filtered detections
    annotated_image = bounding_box_annotator.annotate(scene=frame, detections=high_conf_detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=high_conf_detections)

    cv2.imshow('Webcam', annotated_image)

    k = cv2.waitKey(1)

    if k % 256 == 27:  # ESC key
        print("Escape hit, closing...")
        break

cap.release()
cv2.destroyAllWindows()
