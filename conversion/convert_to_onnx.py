"""
==============================================================================
  STEP 1: Konversi YOLOv8 .pt → .onnx
==============================================================================
  Script ini bisa dijalankan di PC/Laptop (tidak perlu TensorRT).
  Hasil ONNX kemudian dipindahkan ke Jetson Nano untuk konversi ke TensorRT.

  Cara pakai:
      python convert_to_onnx.py
      python convert_to_onnx.py --model /path/to/model.pt
==============================================================================
"""

import os
import sys
import time
import argparse
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] Ultralytics belum terinstall!")
    print("        Jalankan: pip install ultralytics")
    sys.exit(1)


# ============================================================================
#  KONFIGURASI DEFAULT
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

# Default model path (relative to repo root)
DEFAULT_MODEL_PATH = os.path.join(ROOT_DIR, "models", "best_yolov8n.pt")

# Folder output
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "convert_result")

# Ukuran input (sesuaikan dengan training)
IMG_SIZE = 640

# Opset ONNX (11-13 biasanya kompatibel dengan Jetson Nano)
ONNX_OPSET = 11

# Simplify ONNX model (opsional, untuk optimisasi)
SIMPLIFY = True


# ============================================================================
#  PROSES KONVERSI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Konversi YOLOv8 .pt → .onnx")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH,
                        help=f"Path ke model .pt (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--imgsz", type=int, default=IMG_SIZE,
                        help=f"Ukuran input (default: {IMG_SIZE})")
    parser.add_argument("--opset", type=int, default=ONNX_OPSET,
                        help=f"ONNX opset version (default: {ONNX_OPSET})")
    args = parser.parse_args()

    model_path = args.model

    print("=" * 70)
    print("  KONVERSI YOLOv8: PyTorch (.pt) → ONNX (.onnx)")
    print("  Target Deployment: NVIDIA Jetson Nano")
    print("=" * 70)

    # Validasi model
    if not os.path.exists(model_path):
        print(f"\n[ERROR] Model tidak ditemukan: {model_path}")
        print(f"        Pastikan model .pt ada di folder models/")
        sys.exit(1)

    # Buat folder output
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n[INFO] Model input  : {model_path}")
    print(f"[INFO] Output folder: {OUTPUT_DIR}")
    print(f"[INFO] Image size   : {args.imgsz}x{args.imgsz}")
    print(f"[INFO] ONNX Opset   : {args.opset}")
    print(f"[INFO] Simplify     : {SIMPLIFY}")

    # Load model
    print("\n[1/3] Loading model...")
    model = YOLO(model_path)

    # Tampilkan info model
    print(f"       - Task : {model.task}")
    print(f"       - Names: {model.names}")

    # Export ke ONNX
    print("\n[2/3] Mengekspor ke format ONNX...")
    start_time = time.time()

    export_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=SIMPLIFY,
        dynamic=False,       # Static shape untuk Jetson Nano (lebih optimal)
        half=False,          # FP32 ONNX, FP16 dilakukan saat konversi TensorRT
    )

    elapsed = time.time() - start_time
    print(f"       Export selesai dalam {elapsed:.1f} detik")

    # Pindahkan file ONNX ke folder output
    print("\n[3/3] Memindahkan file ke folder output...")

    if export_path and os.path.exists(export_path):
        import shutil
        dest_path = os.path.join(OUTPUT_DIR, os.path.basename(export_path))
        shutil.move(str(export_path), dest_path)

        file_size = os.path.getsize(dest_path) / (1024 * 1024)  # MB
        print(f"       File ONNX: {dest_path}")
        print(f"       Ukuran   : {file_size:.2f} MB")
    else:
        # Cek apakah file ONNX ada di lokasi model
        onnx_path = model_path.replace(".pt", ".onnx")
        if os.path.exists(onnx_path):
            import shutil
            dest_path = os.path.join(OUTPUT_DIR, os.path.basename(onnx_path))
            shutil.move(onnx_path, dest_path)
            file_size = os.path.getsize(dest_path) / (1024 * 1024)
            print(f"       File ONNX: {dest_path}")
            print(f"       Ukuran   : {file_size:.2f} MB")
        else:
            print("[WARNING] File ONNX tidak ditemukan di lokasi yang diharapkan.")
            print("          Cek folder model untuk file .onnx")

    # Instruksi selanjutnya
    print("\n" + "=" * 70)
    print("  LANGKAH SELANJUTNYA:")
    print("=" * 70)
    print("""
  1. Transfer file .onnx ke Jetson Nano:
     scp convert_result/best.onnx user@<jetson-ip>:/path/to/conversion/convert_result/

  2. Di Jetson Nano, jalankan:
     python3 convert_onnx_to_trt.py

  3. File .engine akan tersimpan di folder convert_result/
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()
