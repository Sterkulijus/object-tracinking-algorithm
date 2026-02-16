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

python -m src.main --dataset_path <path> --model yolov8n.pt

## Outputs

- annotated.mp4
- summary.json
- dwell_times.json

## Known Limitations
- FPS must match real video FPS
- Filename sorting not natural
- Center-based crossing may mis-trigger
