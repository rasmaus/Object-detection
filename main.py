from ultralytics import YOLO
import cv2
from PIL import Image

model = YOLO("yolov8l.pt")
results = model("Screenshot 2025-06-21 013910.png")

# Save the result image
results[0].save(filename='output3.jpg')


