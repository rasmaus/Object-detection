# YOLO with CPU and Video Saving

from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import threading
import winsound
import tkinter as tk
from tkinter import messagebox

# Load video
cap = cv2.VideoCapture("0705.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
wait = int(1000 / fps) if fps > 0 else 30

# Get frame dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set up output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("analyzed_output.mp4", fourcc, fps, (frame_width, frame_height))

model = YOLO("yolov8l.pt")

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

low_stock_threshold = 2
low_stock_alerted = False
show_popup = False

bottle_history = []
max_history_time = 60  # seconds

def show_alert_popup():
    root = tk.Tk()
    root.withdraw()
    messagebox.showwarning("Low Stock Alert", "ALERT: Stock is LOW! Please refill bottles.")
    root.destroy()

def play_alert_sound():
    winsound.Beep(1000, 500)

def calculate_selling_rate(bottle_history):
    current_time = time.time()
    while bottle_history and current_time - bottle_history[0][0] > max_history_time:
        bottle_history.pop(0)
    if len(bottle_history) < 2:
        return 0
    first_time, first_count = bottle_history[0]
    last_time, last_count = bottle_history[-1]
    time_diff = (last_time - first_time) / 60
    if time_diff == 0:
        return 0
    count_diff = first_count - last_count
    return max(0, count_diff / time_diff)

while True:
    success, img = cap.read()
    if not success:
        break

    current_time = time.time()
    results = model(img, stream=True)
    bottle_count = 0

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            if cls < len(classNames):
                currentClass = classNames[cls]
            else:
                continue  # Skip invalid class indices

            if currentClass == "bottle" and conf > 0.5:
                bottle_count += 1
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cvzone.putTextRect(img, f"{currentClass} {conf}", (x1, max(y1 - 10, 35)),
                                   scale=1, thickness=1, offset=3)

    if not bottle_history or current_time - bottle_history[-1][0] >= 1:
        bottle_history.append((current_time, bottle_count))

    selling_rate = calculate_selling_rate(bottle_history)

    cvzone.putTextRect(img, f"Bottle Count: {bottle_count}", (50, 50), scale=2, thickness=2, offset=5)
    cvzone.putTextRect(img, f"Selling Rate: {selling_rate:.1f} bottles/min", (50, 150), scale=2, thickness=2, offset=5)

    if bottle_count < low_stock_threshold:
        cvzone.putTextRect(img, "ALERT: Stock is LOW! Please refill bottles.",
                           (50, 100), scale=2, thickness=2, colorR=(0, 0, 255), offset=10)

        if not low_stock_alerted:
            low_stock_alerted = True
            play_alert_sound()
            if not show_popup:
                show_popup = True
                popup_thread = threading.Thread(target=show_alert_popup)
                popup_thread.daemon = True
                popup_thread.start()
    else:
        low_stock_alerted = False
        show_popup = False

    # Show frame
    cv2.imshow("Image", img)

    # Write frame to output video
    out.write(img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

