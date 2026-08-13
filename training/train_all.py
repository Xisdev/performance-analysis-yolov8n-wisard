"""
==============================================================================
  Train ALL - Jalankan Semua Varian YOLOv8 Secara Berurutan
==============================================================================
  Script ini menjalankan training untuk semua 5 varian YOLOv8 (n, s, m, l, x)
  secara berurutan dalam satu kali eksekusi.

  Dimulai dari model paling kecil (n) ke paling besar (x).
  Jika satu model gagal (misal OOM), script akan lanjut ke model berikutnya.

  Cara pakai:
      python train_all.py                   # Training semua (n, s, m, l, x)
      python train_all.py --skip n s        # Skip n dan s, mulai dari m
      python train_all.py --only m l        # Hanya training m dan l

  Hyperparameter:
      Semua hyperparameter terkunci di config.py.
      Epochs: 400 | Early Stopping patience: 100 | Optimizer: AdamW

  Estimasi total waktu: ~62-89 jam di RTX 3060 12GB
==============================================================================
"""

import os
import sys
import time
import argparse
import traceback
import torch
from ultralytics import YOLO
from multiprocessing import freeze_support

# cuDNN: benchmark=True otomatis pilih algoritma konvolusi tercepat
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Import konfigurasi terpusat
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    TRAINING_CONFIG, BATCH_SIZE, DATA_YAML,
    PROJECT_DIR, COLLECTED_DIR, DATASET_NAME,
    copy_best_model
)

ALL_VARIANTS = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']


def train_single(variant):
    """Training satu varian model."""
    run_name = f'{variant}_{DATASET_NAME}'
    batch = BATCH_SIZE[variant]

    print(f"\n{'=' * 60}")
    print(f"  TRAINING: {variant.upper()}")
    print(f"  Optimizer: {TRAINING_CONFIG['optimizer']} | Batch: {batch}")
    print(f"  Epochs: {TRAINING_CONFIG['epochs']} | Patience: {TRAINING_CONFIG['patience']}")
    print(f"  Dataset: {DATASET_NAME} (31554 train / 16302 val)")
    print(f"{'=' * 60}")

    torch.cuda.empty_cache()

    model = YOLO(f'{variant}.pt')

    t_start = time.time()

    results = model.train(
        data=DATA_YAML,
        batch=batch,
        project=PROJECT_DIR,
        name=run_name,
        **TRAINING_CONFIG,
    )

    t_elapsed = time.time() - t_start
    hours = t_elapsed / 3600

    # Salin weights: folder asli TETAP, + folder bernama + folder terpusat
    print(f"\n--- Menyalin weights ---")
    best_dst = copy_best_model(variant)

    print(f"\n[OK] Training {variant.upper()} selesai! ({hours:.1f} jam)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Training semua varian YOLOv8 (n/s/m/l/x)"
    )
    parser.add_argument(
        '--skip', nargs='+', default=[],
        help="Varian yang di-skip (misal: --skip n s)"
    )
    parser.add_argument(
        '--only', nargs='+', default=[],
        help="Hanya training varian tertentu (misal: --only m l x)"
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA tidak tersedia!")
        return

    print(f"{'=' * 60}")
    print(f"  TRAIN ALL - YOLOv8 (n/s/m/l/x)")
    print(f"  GPU      : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  Optimizer: {TRAINING_CONFIG['optimizer']}")
    print(f"  LR       : {TRAINING_CONFIG['lr0']} (wd={TRAINING_CONFIG['weight_decay']})")
    print(f"  Epochs   : {TRAINING_CONFIG['epochs']} (patience={TRAINING_CONFIG['patience']})")
    print(f"{'=' * 60}")

    # Tentukan varian yang akan di-train
    if args.only:
        variants = [f'yolov8{v}' for v in args.only]
    else:
        skip_set = {f'yolov8{v}' for v in args.skip}
        variants = [v for v in ALL_VARIANTS if v not in skip_set]

    print(f"\nVarian  : {', '.join([v.upper() for v in variants])}")
    print(f"Batch   : {', '.join([f'{v}={BATCH_SIZE[v]}' for v in variants])}")
    print(f"Output  : {PROJECT_DIR}")

    results = {}
    t_total_start = time.time()

    for i, variant in enumerate(variants):
        print(f"\n{'#' * 60}")
        print(f"  [{i+1}/{len(variants)}] {variant.upper()}")
        print(f"{'#' * 60}")

        try:
            success = train_single(variant)
            results[variant] = 'OK' if success else 'FAILED'
        except torch.cuda.OutOfMemoryError:
            print(f"\n[ERROR] {variant.upper()} - OUT OF MEMORY!")
            print(f"        Coba turunkan batch size di config.py")
            results[variant] = 'OOM'
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"\n[ERROR] {variant.upper()} - {e}")
            traceback.print_exc()
            results[variant] = f'ERROR: {str(e)[:50]}'
            torch.cuda.empty_cache()

    t_total = (time.time() - t_total_start) / 3600

    # Ringkasan
    print(f"\n{'=' * 60}")
    print(f"  RINGKASAN TRAINING")
    print(f"  Total waktu: {t_total:.1f} jam")
    print(f"  Optimizer  : {TRAINING_CONFIG['optimizer']}")
    print(f"  Epochs     : {TRAINING_CONFIG['epochs']} (patience={TRAINING_CONFIG['patience']})")
    print(f"{'=' * 60}")
    for variant, status in results.items():
        icon = "[OK]" if status == 'OK' else "[!!]"
        print(f"  {icon} {variant.upper():10s} - {status}")

    # Cek model yang terkumpul
    if os.path.exists(COLLECTED_DIR):
        models = [f for f in os.listdir(COLLECTED_DIR) if f.endswith('.pt')]
        print(f"\nModel terkumpul di: {COLLECTED_DIR}")
        for m in sorted(models):
            size = os.path.getsize(os.path.join(COLLECTED_DIR, m)) / (1024*1024)
            print(f"  - {m} ({size:.1f} MB)")

    print(f"\n{'=' * 60}")


if __name__ == '__main__':
    freeze_support()
    main()
