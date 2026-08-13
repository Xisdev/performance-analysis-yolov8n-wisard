"""
=============================================================================
  Prepare Test Images for Device Benchmarking
=============================================================================
  Script ini menyiapkan subset gambar test dari dataset WiSARD untuk
  ditransfer ke Jetson Nano dan Raspberry Pi 4.

  Cara pakai:
      python prepare_test_images.py                    # Default 200 gambar
      python prepare_test_images.py --num 100          # 100 gambar
      python prepare_test_images.py --num 0            # Semua gambar test

  Output:
      test_for_device/test_images/
      ├── images/
      │   ├── img_0001.jpg
      │   └── ...
      └── labels/
          ├── img_0001.txt
          └── ...
=============================================================================
"""

import os
import sys
import shutil
import random
import argparse
from pathlib import Path


# ============================================================================
#  KONFIGURASI
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)  # D:\RISET\drone-wisard

# Path dataset WiSARD
DATASET_IMAGE_DIR = os.path.join(ROOT_DIR, "datasets", "wisard_ir", "images", "test")
DATASET_LABEL_DIR = os.path.join(ROOT_DIR, "datasets", "wisard_ir", "labels", "test")

# Output
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "test_images")
OUTPUT_IMG_DIR = os.path.join(OUTPUT_DIR, "images")
OUTPUT_LBL_DIR = os.path.join(OUTPUT_DIR, "labels")

# Default jumlah gambar
DEFAULT_NUM_IMAGES = 200

# Seed untuk reproducibility
RANDOM_SEED = 42

# Ekstensi gambar yang didukung
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ============================================================================
#  FUNGSI UTAMA
# ============================================================================

def get_image_files(image_dir):
    """Ambil semua file gambar dari direktori."""
    files = []
    for f in sorted(os.listdir(image_dir)):
        if Path(f).suffix.lower() in IMG_EXTENSIONS:
            files.append(f)
    return files


def prepare_images(num_images=DEFAULT_NUM_IMAGES, seed=RANDOM_SEED):
    """Siapkan subset gambar test untuk benchmarking."""

    print("=" * 70)
    print("  PERSIAPAN GAMBAR TEST UNTUK DEVICE BENCHMARKING")
    print("=" * 70)

    # Validasi input
    if not os.path.isdir(DATASET_IMAGE_DIR):
        print(f"\n[ERROR] Folder gambar test tidak ditemukan:")
        print(f"        {DATASET_IMAGE_DIR}")
        sys.exit(1)

    # Ambil semua gambar
    all_images = get_image_files(DATASET_IMAGE_DIR)
    total_available = len(all_images)
    print(f"\n[INFO] Dataset test  : {DATASET_IMAGE_DIR}")
    print(f"[INFO] Total gambar  : {total_available}")

    if total_available == 0:
        print("[ERROR] Tidak ada gambar ditemukan di folder test!")
        sys.exit(1)

    # Tentukan jumlah
    if num_images <= 0 or num_images >= total_available:
        selected = all_images
        print(f"[INFO] Menggunakan SEMUA gambar ({total_available})")
    else:
        random.seed(seed)
        selected = sorted(random.sample(all_images, num_images))
        print(f"[INFO] Memilih {num_images} gambar acak (seed={seed})")

    # Buat folder output
    if os.path.exists(OUTPUT_DIR):
        print(f"\n[WARN] Folder output sudah ada, menghapus...")
        shutil.rmtree(OUTPUT_DIR)

    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LBL_DIR, exist_ok=True)

    # Copy gambar dan label
    print(f"\n[INFO] Menyalin {len(selected)} gambar + label...")
    copied_img = 0
    copied_lbl = 0
    missing_lbl = 0

    for i, img_file in enumerate(selected):
        # Copy gambar
        src_img = os.path.join(DATASET_IMAGE_DIR, img_file)
        dst_img = os.path.join(OUTPUT_IMG_DIR, img_file)
        shutil.copy2(src_img, dst_img)
        copied_img += 1

        # Copy label (jika ada)
        lbl_file = Path(img_file).stem + ".txt"
        src_lbl = os.path.join(DATASET_LABEL_DIR, lbl_file)
        if os.path.exists(src_lbl):
            dst_lbl = os.path.join(OUTPUT_LBL_DIR, lbl_file)
            shutil.copy2(src_lbl, dst_lbl)
            copied_lbl += 1
        else:
            missing_lbl += 1

        # Progress
        if (i + 1) % 50 == 0 or (i + 1) == len(selected):
            print(f"       {i + 1}/{len(selected)} file disalin...")

    # Hitung ukuran folder
    total_size = 0
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for f in files:
            total_size += os.path.getsize(os.path.join(root, f))
    size_mb = total_size / (1024 * 1024)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  SELESAI")
    print(f"{'=' * 70}")
    print(f"  Gambar disalin  : {copied_img}")
    print(f"  Label disalin   : {copied_lbl}")
    print(f"  Label hilang    : {missing_lbl}")
    print(f"  Total ukuran    : {size_mb:.1f} MB")
    print(f"  Output folder   : {OUTPUT_DIR}")
    print(f"{'=' * 70}")

    print(f"\n  Langkah selanjutnya:")
    print(f"  1. Transfer folder 'test_images' ke Jetson Nano:")
    print(f"     scp -r test_images/ user@<jetson-ip>:~/benchmark/")
    print(f"")
    print(f"  2. Transfer folder 'test_images' ke Raspberry Pi 4:")
    print(f"     scp -r test_images/ pi@<raspi-ip>:~/benchmark/")
    print(f"{'=' * 70}")


# ============================================================================
#  ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Siapkan subset gambar test WiSARD untuk device benchmarking"
    )
    parser.add_argument(
        "--num", "-n", type=int, default=DEFAULT_NUM_IMAGES,
        help=f"Jumlah gambar yang akan dipilih (default: {DEFAULT_NUM_IMAGES}, 0=semua)"
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=RANDOM_SEED,
        help=f"Random seed untuk reproducibility (default: {RANDOM_SEED})"
    )
    args = parser.parse_args()

    prepare_images(num_images=args.num, seed=args.seed)
