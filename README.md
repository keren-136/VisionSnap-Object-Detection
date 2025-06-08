# 📸 VisionSnap – Assistive Object Detection for the Visually Impaired

VisionSnap is a Python-based assistive tool designed to help visually impaired users identify whether an object is positioned in front of them, using real-time object detection and localization.

---

## 💡 What It Does

- 🧠 Detects real-world objects using a webcam
- 🎯 Classifies object location as "Left", "Center", or "Right"
- 🔊 (Optional) Provides auditory feedback using text-to-speech
- ⚡ Fast, lightweight, and runs in real-time

---

## 🛠️ Tech Stack

- **Language**: Python 3.x
- **Libraries**: 
  - OpenCV (video processing)
  - Ultralytics YOLOv5 / YOLOv8 (object detection)
  - PyTorch (model backend)
  - NumPy
  - gTTS / pyttsx3 (for audio feedback)

---


## ⚙️ How to Run

1. Clone the repo  
   `git clone https://github.com/yourusername/VisionSnap-Object-Detection.git`

2. Install dependencies  
   ```bash
   pip install -r requirements.txt
