#yollo with CPU

from ultralytics import YOLO
import cv2
import cvzone 
import math

#fo the web cam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

#for the video
#cap = cv2.VideoCapture("27268-363287559_medium.mp4")
#fps = cap.get(cv2.CAP_PROP_FPS)
#wait = int(1000 / fps)  # Delay between frames in milliseconds



model = YOLO("best (1).pt")

# Full list of 80 COCO classes
classNames = ['Beans', 'Cake','Candy', 'Cereal',
              'Chips', 'Chocolate', 'Coffee', 'Corn', 
              'Fish', 'Flour', 'Honey', 'Jam', 
              'Juice','Milk', 'Nuts', 'Oil',
              'Pasta','Rice','Soda', 'Spices',
              'Sugar', 'Tea', 'Tomato Sauce', 'Vinegar', 
              'Water']

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            w,h=x2-x1, y2-y1
            cvzone.cornerRect(img, (x1, y1, w, h))   
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            cvzone.putTextRect(img, f"{currentClass} {conf}", (max(0, x1), max(35, y1)), scale=1, thickness=1, offset=3)
           
                
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    #if cv2.waitKey(wait) & 0xFF == ord('q'):
     #   break
