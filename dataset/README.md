# 📦 WiSARD IR Dataset

## Overview

The **WiSARD IR (Wilderness Search and Rescue Dataset — Infrared)** is a thermal infrared image dataset designed for human detection in search and rescue scenarios using drones.

## Dataset Statistics

| Split | Images | Percentage |
|-------|--------|-----------|
| Training | 31,554 | 58.6% |
| Validation | 16,302 | 30.3% |
| Testing | 5,950 | 11.1% |
| **Total** | **53,806** | **100%** |

## Classes

| Class ID | Name |
|----------|------|
| 0 | human |

## Format

The dataset follows the **YOLO format**:

```
dataset/
├── data.yaml           # Dataset configuration
├── images/
│   ├── train/          # Training images
│   ├── val/            # Validation images
│   └── test/           # Test images
└── labels/
    ├── train/          # Training labels (YOLO txt format)
    ├── val/            # Validation labels
    └── test/           # Test labels
```

Each label file contains one line per object:
```
<class_id> <x_center> <y_center> <width> <height>
```
All coordinates are normalized to [0, 1].

## Sample Images

Sample thermal infrared images from the dataset are included in `sample_images/`:

| Prefix | Split |
|--------|-------|
| `train_*` | Training set |
| `val_*` | Validation set |
| `test_*` | Test set |

## Download

> ⚠️ **The full dataset (~8 GB) is not included in this repository.**

To obtain the full dataset:

1. Download the WiSARD IR dataset from [https://drive.google.com/file/d/1PKjGCqUszHH1nMbXUBTwPSDqRabAt_ht/view]
2. Extract the archive
3. Place the `images/` and `labels/` directories alongside `data.yaml`:

```bash
# Expected structure after download
dataset/
├── data.yaml          # Already included in repo
├── images/
│   ├── train/         # 31,554 images
│   ├── val/           # 16,302 images
│   └── test/          # 5,950 images
└── labels/
    ├── train/
    ├── val/
    └── test/
```

## data.yaml Configuration

```yaml
train: images/train
val: images/val
test: images/test

names:
  0: human
```

## Image Characteristics

- **Type**: Thermal infrared (grayscale)
- **Source**: Drone-mounted FLIR thermal cameras
- **Scenarios**: Wilderness, forest, open terrain
- **Conditions**: Day/night, various altitudes, occluded subjects
