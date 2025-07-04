# YOLO with CPU

from ultralytics import YOLO
import cv2
import cvzone
import math

# For the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("yolov8n.pt")

# Full list of 80 COCO classes
classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# Set bottle low-stock threshold
low_stock_threshold = 2

while True:
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True)
    bottle_count = 0  # Reset bottle count per frame

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.ceil((box.conf[0] * 100)) / 100

            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Only track bottles with confidence > 0.5
            if currentClass == "bottle" and conf > 0.5:
                bottle_count += 1
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cvzone.putTextRect(img, f"{currentClass} {conf}", (x1, max(y1 - 10, 35)), scale=1, thickness=1, offset=3)

    # Display bottle count
    cvzone.putTextRect(img, f"Bottle Count: {bottle_count}", (50, 50), scale=2, thickness=2, offset=5)

    # Show alert if bottle count is low
    if bottle_count < low_stock_threshold:
        cvzone.putTextRect(
            img,
            "ALERT: Stock is LOW! Please refill bottles.",
            (50, 100),
            scale=2,
            thickness=2,
            colorR=(0, 0, 255),
            offset=10
        )

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
