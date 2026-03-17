from ultralytics import YOLO
import cv2
import torch
import time

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Select device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Get webcam frame size
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
out = cv2.VideoWriter("Video.mp4", fourcc, 24, (width, height))

# FPS variables
prev_time = 0

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Error: Could not read frame.")
        break

    # Run YOLO detection
    results = model(img, imgsz=320, conf=0.5, device=device)

    # Draw detections
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = int(box.conf[0] * 100)
            cls = int(box.cls[0])
            label = model.names[cls]

            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Draw label and confidence
            cv2.putText(
                img,
                f"{label} {conf}%",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    # Show FPS
    cv2.putText(
        img,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Show device
    cv2.putText(
        img,
        f"Device: {device}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Write frame to video
    out.write(img)

    # Show output
    cv2.imshow("YOLOv8 Object Detection", img)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()