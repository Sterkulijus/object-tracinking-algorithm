# Object Detection and Line-Cross Counting

## Overview
Trumpas projekto tikslas:
YOLOv8 + ByteTrack + line-cross event detection + dwell time analysis.

## Pipeline
1. Detection (YOLOv8)
2. Tracking (ByteTrack)
3. Line-cross counting
4. Dwell time computation
5. Annotated video + JSON outputs

## Installation

pip install -r requirements.txt

## Usage

### Run (auto download + process first found sequence):
  python ua_detrac_yolo_track_count.py --download

### Run (use existing downloaded path):
  python ua_detrac_yolo_track_count.py --dataset_path "YOUR_PATH_FROM_kagglehub" --sequence_auto
  
### Run (choose a specific sequence folder name substring):
  python ua_detrac_yolo_track_count.py --dataset_path "..." --sequence_hint "MVI_20011"

## Outputs

- annotated.mp4
- summary.json
- dwell_times.json

## Known Limitations
- FPS must match real video FPS
- Filename sorting not natural
- Center-based crossing may mis-trigger
