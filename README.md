# Hand Gesture Virtual Whiteboard

A real-time **AI Hand Gesture Whiteboard** built with **Python, OpenCV, and MediaPipe** that allows users to draw, highlight, erase, and control a whiteboard using only hand gestures captured by a webcam.

---

# Demo

https://github.com/user-attachments/assets/760d43b1-0d31-4f17-bfcf-fc108b792f3e

---

# Features

• Real-time hand tracking using MediaPipe  
• Gesture-based drawing interface  
• Smooth and low-latency drawing  
• Highlighter and pen tools  
• Whole-hand eraser for faster cleaning  
• Whiteboard background mode  
• Gesture stabilization to prevent accidental triggers  
• Finger joint detection and connection lines  
• Anti-mess drawing logic to prevent stray strokes  

---

# Gesture Controls

| Gesture | Action |
|-------|-------|
| ☝️ Index finger | Draw |
| ✌️ Two fingers | Select blue pen |
| 🤟 Three fingers | Select green pen |
| 🖖 Four fingers | Highlighter |
| ✋ Five fingers | Eraser |
| 🙌 Both hands open | Toggle whiteboard |

---

# Tech Stack

Python  
OpenCV  
MediaPipe Hands  
NumPy

---

# Project Structure

HandGestureWhiteboard

app.py  
hand_tracker.py  
gesture_logic.py  
drawing_canvas.py  
requirements.txt  
demo.mp4  
README.md

---

# Installation

Clone the repository

git clone https://github.com/yourusername/HandGestureWhiteboard.git

cd HandGestureWhiteboard

Install dependencies

pip install -r requirements.txt

---

# Run the Project

python app.py

Your webcam will open and the gesture whiteboard will start.

Press **Q** or **ESC** to exit.

---

# System Requirements

Minimum tested configuration

CPU: Intel Core i7 8th Gen  
RAM: 8GB  
Python: 3.9+  
Webcam

GPU is **not required**.

---

# How It Works

### Hand Detection
MediaPipe Hands detects **21 keypoints for each hand** in real time.

### Finger State Detection
Finger positions determine whether each finger is **up or down**, forming a gesture vector.

Example

[thumb, index, middle, ring, pinky]

### Gesture Classification

Finger combinations are mapped to actions

[0,1,0,0,0] → Draw  
[0,1,1,0,0] → Blue pen  
[0,1,1,1,0] → Green pen  
[0,1,1,1,1] → Highlighter  
[1,1,1,1,1] → Eraser

### Gesture Stabilization

Gestures are stabilized across multiple frames to avoid accidental triggers and flickering actions.

### Drawing Engine

The drawing engine uses

• smoothing filters for stable cursor movement  
• interpolation between points to avoid gaps  
• layered canvas for pen and highlighter  
• rectangle-based eraser for fast clearing

---

# Optimization Techniques

The system is optimized for CPU performance using

• reduced MediaPipe model complexity  
• frame scaling before inference  
• One-Euro filter smoothing  
• gesture stabilization buffers  
• interpolated drawing strokes

---

# Future Improvements

Possible improvements include

• gesture-based undo / redo  
• shape recognition  
• handwriting recognition  
• color palette UI  
• export drawings as image or PDF  
• collaborative multi-user whiteboard

---

# Author

Munad E Ali

---

# License

This project is open source under the **MIT License**.
