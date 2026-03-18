# Real-Time Object Detection using YOLOv8

## Overview

This project implements a real-time object detection system using the YOLOv8 (You Only Look Once) model. The application captures live video from a webcam, performs object detection, and displays bounding boxes with class labels and confidence scores.

The system is designed to demonstrate practical applications of computer vision and deep learning, with a focus on performance optimization and real-time processing.

---

## Features

* Real-time object detection using YOLOv8
* Live webcam video processing
* Bounding boxes with class labels and confidence scores
* Frames Per Second (FPS) monitoring
* Video recording of detection output
* Support for GPU acceleration (CUDA)

---

## Technologies Used

* Python
* OpenCV (cv2)
* PyTorch
* Ultralytics YOLOv8

---

## Installation

Install the required dependencies:

```bash
pip install ultralytics opencv-python torch
```

---

## Usage

Run the application:

```bash
python main.py
```

Press `ESC` or `Q` to terminate the program.

---

## Project Structure

```
object-detection/
│── main.py
│── Video.mp4
│── README.md
```

---

## Performance Considerations

* The system supports GPU acceleration for improved inference speed.
* Image resolution and confidence thresholds can be adjusted to balance accuracy and performance.
* On CPU, lower image sizes are recommended for real-time responsiveness.

---

## Future Improvements

* Custom object detection model training
* Integration with web or mobile interfaces
* Object tracking across frames
* Deployment as a cloud-based service

---

## Author

Ayush Sainju
