"""
==============================================================================
  Training YOLOv8x - Extra Large (Paling Berat)
==============================================================================
  Varian terbesar dan paling akurat, tapi paling lambat.
  Diperkirakan akan OOM di Raspberry Pi 4 saat benchmark nanti.
  Estimasi waktu: ~24-36 jam di RTX 3060 (400 epoch, batch=4)
  Estimasi VRAM: ~10 GB (KETAT, hampir mentok 12GB)

  Semua hyperparameter terkunci di config.py (JANGAN ubah di sini).

  CATATAN: Jika terjadi OOM saat training, turunkan batch ke 2:
    Ubah BATCH_SIZE['yolov8x'] = 2 di config.py
==============================================================================
"""
import os
import sys
import torch
from ultralytics import YOLO
from multiprocessing import freeze_support

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    TRAINING_CONFIG, BATCH_SIZE, DATA_YAML,
    PROJECT_DIR, DATASET_NAME, copy_best_model
)

MODEL_VARIANT = 'yolov8x'
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
