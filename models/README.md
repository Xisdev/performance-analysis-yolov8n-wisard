# 📥 Pre-trained Model Weights

## Overview

This directory is intended to store pre-trained YOLOv8 model weights trained on the WiSARD IR thermal infrared dataset.

> ⚠️ **Model weights are NOT included in this repository** due to file size constraints.

## Available Models

| Model | Parameters | mAP50 | File Size | Format |
|-------|-----------|-------|-----------|--------|
| YOLOv8n | 3.2M | 53.20% | 6 MB | `.pt` |
| YOLOv8s | 11.2M | 51.30% | 22 MB | `.pt` |
| YOLOv8m | 25.9M | 49.48% | 52 MB | `.pt` |
| YOLOv8l | 43.7M | 57.92% | 87 MB | `.pt` |
| YOLOv8x | 68.2M | 53.63% | 137 MB | `.pt` |

## Download

Download pre-trained weights from: **[https://drive.google.com/drive/folders/1KLwcvxjbaATgcCAOZ6mj9RvNxdj5eP0O?usp=sharing]**

<!-- 
TODO: Update with actual download links
Options:
- Google Drive
- Hugging Face Hub
- GitHub Releases (for files < 2GB)
-->

## Setup

After downloading, place the weights in this directory:

```
models/
├── best_yolov8n.pt
├── best_yolov8s.pt
├── best_yolov8m.pt
├── best_yolov8l.pt
└── best_yolov8x.pt
```

## Usage

```python
from ultralytics import YOLO

# Load a pre-trained model
model = YOLO("models/best_yolov8n.pt")

# Run inference
results = model.predict(source="image.jpg", conf=0.35)

# Run validation
results = model.val(data="dataset/data.yaml", split="test")
```

## Converted Models

For edge device deployment, you can also convert models to:

| Format | Target Device | Precision | Script |
|--------|--------------|-----------|--------|
| `.onnx` | Cross-platform | FP32 | `conversion/convert_to_onnx.py` |
| `.engine` | Jetson Nano | FP16 | `conversion/convert_onnx_to_trt.py` |
| `.tflite` | Raspberry Pi 4 | FP32 | `conversion/convert_to_tflite.py` |

## Training Your Own

To train from scratch, see the [`training/`](../training/) directory.
