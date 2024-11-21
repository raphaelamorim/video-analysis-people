import cv2
import numpy as np
from ultralytics import YOLO  # For person detection

def detect_people():
    # Initialize the YOLO model
    model = YOLO('yolov8n.pt')  # Using YOLOv8 nano model
    
    # Initialize video capture (0 for webcam, or provide video file path)
    cap = cv2.VideoCapture(0)
    
    while True:
        # Read frame from video/webcam
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run detection on frame
        results = model(frame, classes=[0])  # class 0 is person in COCO dataset
        
        # Process each detection
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Draw rectangle around person
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add confidence score
                conf = float(box.conf[0])
                cv2.putText(frame, f'Person: {conf:.2f}', 
                          (x1, y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          0.5, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Person Detection', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_people()