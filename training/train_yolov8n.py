"""
==============================================================================
  Training YOLOv8n - Nano (Paling Ringan)
==============================================================================
  Varian paling kecil dan cepat. Ideal untuk edge device.
  Estimasi waktu: ~3-5 jam di RTX 3060 (300 epoch, batch=32)
  Estimasi VRAM: ~7 GB
==============================================================================
"""
import os
import sys
import torch
from ultralytics import YOLO
from multiprocessing import freeze_support

# CUDA debugging
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Import konfigurasi terpusat
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    TRAINING_CONFIG, BATCH_SIZE, DATA_YAML,
    PROJECT_DIR, DATASET_NAME, copy_best_model
)

MODEL_VARIANT = 'yolov8n'
RUN_NAME = f'{MODEL_VARIANT}_{DATASET_NAME}'


def jalankan_training():
    if not torch.cuda.is_available():
        print("ERROR: CUDA tidak tersedia!")
        return

    print(f"{'=' * 60}")
    print(f"  TRAINING: {MODEL_VARIANT.upper()}")
    print(f"  Optimizer: {TRAINING_CONFIG['optimizer']} | Batch: {BATCH_SIZE[MODEL_VARIANT]}")
    print(f"  Epochs: {TRAINING_CONFIG['epochs']} | Patience: {TRAINING_CONFIG['patience']}")
    print(f"{'=' * 60}")
    print(f"PyTorch : {torch.__version__}")
    print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"cuDNN   : {torch.backends.cudnn.version()}")

    torch.cuda.empty_cache()
    model = YOLO(f'{MODEL_VARIANT}.pt')

    results = model.train(
        data=DATA_YAML,
        batch=BATCH_SIZE[MODEL_VARIANT],
        project=PROJECT_DIR,
        name=RUN_NAME,
        **TRAINING_CONFIG,
    )

    print(f"\n--- Menyalin weights ---")
    best_dst = copy_best_model(MODEL_VARIANT)

    print(f"\nTraining {MODEL_VARIANT.upper()} selesai!")
    print(f"Hasil   : {os.path.join(PROJECT_DIR, RUN_NAME)}")
    print(f"Best.pt : {best_dst}")


if __name__ == '__main__':
    freeze_support()
    jalankan_training()
