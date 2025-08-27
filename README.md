# AlertDrive: Driver Drowsiness & Distraction Detection

Real-time computer vision system that detects **drowsiness (eye closure, yawning)** and **distraction (looking away)** from a webcam feed. 
Runs on CPU in real-time using **OpenCV + MediaPipe** with simple rule-based thresholds (EAR/MAR/head pose).



---

## Features
- Face & landmark detection via MediaPipe Face Mesh (478 landmarks)
- **EAR** (Eye Aspect Ratio) → blinks & prolonged closures
- **MAR** (Mouth Aspect Ratio) → yawning
- **Head Pose Proxy** using facial landmarks → distraction (looking away / down)
- On-screen alerts + system beep
- Configurable thresholds and frame windows

## Quick Start

### 1) Create & activate a virtual environment (recommended)
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Run
```bash
python src/main.py --cam 0
```

Press **q** to quit.

---

## Configuration (CLI Flags)
- `--cam` : camera index (default: 0)
- `--ear_thresh` : eye aspect ratio threshold (default: 0.21)
- `--ear_frames` : consecutive frames below EAR to trigger drowsiness (default: 12)
- `--mar_thresh` : mouth aspect ratio (yawn) threshold (default: 0.6)
- `--mar_frames` : consecutive frames above MAR to trigger yawn (default: 15)
- `--yaw_thresh` : head yaw (left-right) threshold in normalized coords (default: 0.12)
- `--pitch_thresh` : head pitch (up-down) threshold (default: 0.10)
- `--look_frames` : frames to flag distraction (default: 15)

---

## How it works (brief)
- **Landmarks** from MediaPipe give locations for eyes & mouth.
- **EAR** ≈ vertical eye distance / horizontal eye distance.
- **MAR** ≈ vertical mouth distance / horizontal mouth distance.
- **Head Pose Proxy** uses relative positions of nose vs eyes/ears to estimate left/right/up/down look.
- We maintain counters across frames; when a counter exceeds its window, we raise an alert.

---

