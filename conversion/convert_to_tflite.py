"""
==============================================================================
  Konversi YOLOv8 .pt → .tflite (TensorFlow Lite)
==============================================================================
  Target deployment: Raspberry Pi 4 Model B

  Script ini mengkonversi model YOLOv8 (.pt) ke format TFLite (.tflite)
  yang bisa dijalankan di Raspberry Pi 4 tanpa GPU.

  Cara pakai:
      # Konversi satu model
      python convert_to_tflite.py --input path/to/best.pt

      # Konversi dengan INT8 quantization
      python convert_to_tflite.py --input path/to/best.pt --int8

      # Konversi semua model dari folder collected
      python convert_to_tflite.py --batch

  Prasyarat:
      pip install ultralytics tensorflow
==============================================================================
"""

import os
import sys
import time
import glob
import argparse
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] Ultralytics belum terinstall!")
    print("        Jalankan: pip install ultralytics")
    sys.exit(1)


# ============================================================================
#  KONFIGURASI
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)  # D:\RISET\drone-wisard

# Folder output
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "convert_result")

# Folder model yang sudah di-train (collected)
COLLECTED_DIR = os.path.join(ROOT_DIR, "runs_optimized", "collected_best_models")

# Ukuran input (harus sama dengan training)
IMG_SIZE = 640


# ============================================================================
#  KONVERSI
# ============================================================================

def convert_single_model(model_path, output_dir, use_int8=False, img_size=IMG_SIZE):
    """
    Konversi satu model .pt ke .tflite menggunakan Ultralytics export.

    Parameters:
        model_path: Path ke file .pt
        output_dir: Folder output
        use_int8:   Gunakan INT8 quantization (lebih kecil, sedikit kurang akurat)
        img_size:   Ukuran input model

    Returns:
        Path ke file .tflite yang dihasilkan, atau None jika gagal
    """
    precision = "int8" if use_int8 else "fp32"
    model_name = Path(model_path).stem

    print(f"\n{'─' * 60}")
    print(f"  Konversi: {model_name}.pt → .tflite ({precision.upper()})")
    print(f"{'─' * 60}")
    print(f"  Input     : {model_path}")
    print(f"  Output dir: {output_dir}")
    print(f"  Image size: {img_size}x{img_size}")
    print(f"  Precision : {precision.upper()}")

    # Validasi
    if not os.path.exists(model_path):
        print(f"  [ERROR] File model tidak ditemukan: {model_path}")
        return None

    # Load model
    print(f"\n  [1/3] Loading model...")
    try:
        model = YOLO(model_path)
        print(f"         Task  : {model.task}")
        print(f"         Names : {model.names}")
    except Exception as e:
        print(f"  [ERROR] Gagal load model: {e}")
        return None

    # Export ke TFLite
    print(f"\n  [2/3] Mengekspor ke format TFLite ({precision.upper()})...")
    start_time = time.time()

    try:
        export_path = model.export(
            format="tflite",
            imgsz=img_size,
            int8=use_int8,
            half=False,        # TFLite tidak mendukung FP16 di CPU
        )
    except Exception as e:
        print(f"  [ERROR] Gagal export ke TFLite: {e}")
        print(f"  [TIP]  Pastikan tensorflow terinstall: pip install tensorflow")
        return None

    elapsed = time.time() - start_time
    print(f"         Export selesai dalam {elapsed:.1f} detik")

    # Pindahkan file TFLite ke folder output
    print(f"\n  [3/3] Memindahkan file ke folder output...")
    os.makedirs(output_dir, exist_ok=True)

    # Cari file tflite yang dihasilkan
    tflite_path = None

    if export_path and os.path.exists(str(export_path)):
        tflite_path = str(export_path)
    else:
        # Cari di sekitar model path
        search_patterns = [
            model_path.replace(".pt", "_saved_model/*.tflite"),
            model_path.replace(".pt", "_float32.tflite"),
            model_path.replace(".pt", "_float16.tflite"),
            model_path.replace(".pt", "_int8.tflite"),
            model_path.replace(".pt", ".tflite"),
        ]
        for pattern in search_patterns:
            matches = glob.glob(pattern)
            if matches:
                tflite_path = matches[0]
                break

        # Cari di folder _saved_model
        saved_model_dir = model_path.replace(".pt", "_saved_model")
        if tflite_path is None and os.path.isdir(saved_model_dir):
            for root, dirs, files in os.walk(saved_model_dir):
                for f in files:
                    if f.endswith(".tflite"):
                        tflite_path = os.path.join(root, f)
                        break
                if tflite_path:
                    break

    if tflite_path and os.path.exists(tflite_path):
        import shutil
        # Nama file output yang jelas
        dest_name = f"{model_name}_{precision}.tflite"
        dest_path = os.path.join(output_dir, dest_name)
        shutil.copy2(tflite_path, dest_path)

        file_size = os.path.getsize(dest_path) / (1024 * 1024)  # MB
        print(f"         File TFLite : {dest_path}")
        print(f"         Ukuran      : {file_size:.2f} MB")

        # Cleanup saved_model folder yang besar
        saved_model_dir = model_path.replace(".pt", "_saved_model")
        if os.path.isdir(saved_model_dir):
            print(f"         Cleanup     : Menghapus folder _saved_model...")
            shutil.rmtree(saved_model_dir, ignore_errors=True)

        return dest_path
    else:
        print(f"  [ERROR] File TFLite tidak ditemukan setelah export!")
        print(f"  [TIP]  Cek manual di folder model untuk file .tflite")
        return None


def convert_batch(model_dir, output_dir, use_int8=False):
    """Konversi semua model .pt di suatu folder."""
    print(f"\n[BATCH] Mencari model .pt di: {model_dir}")

    if not os.path.isdir(model_dir):
        print(f"[ERROR] Folder tidak ditemukan: {model_dir}")
        return

    pt_files = sorted(glob.glob(os.path.join(model_dir, "*.pt")))
    if not pt_files:
        print("[ERROR] Tidak ada file .pt ditemukan!")
        print(f"        Pastikan model sudah di-train dan disalin ke:\n        {model_dir}")
        return

    print(f"[BATCH] Ditemukan {len(pt_files)} model:")
    for f in pt_files:
        print(f"        - {os.path.basename(f)}")

    results = []
    for pt_file in pt_files:
        result = convert_single_model(pt_file, output_dir, use_int8)
        results.append((os.path.basename(pt_file), result))

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  BATCH KONVERSI SELESAI")
    print(f"{'=' * 70}")
    for name, path in results:
        status = "✓" if path else "✗"
        print(f"  {status} {name}")
    print(f"{'=' * 70}")


# ============================================================================
#  MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Konversi YOLOv8 .pt → .tflite untuk Raspberry Pi 4"
    )
    parser.add_argument(
        "--input", "-i", type=str, default=None,
        help="Path ke file .pt tunggal"
    )
    parser.add_argument(
        "--batch", "-b", action="store_true",
        help=f"Konversi semua model di folder collected ({COLLECTED_DIR})"
    )
    parser.add_argument(
        "--model-dir", type=str, default=COLLECTED_DIR,
        help=f"Folder berisi model .pt untuk mode batch (default: {COLLECTED_DIR})"
    )
    parser.add_argument(
        "--int8", action="store_true",
        help="Gunakan INT8 quantization (model lebih kecil, sedikit kurang akurat)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=IMG_SIZE,
        help=f"Ukuran input model (default: {IMG_SIZE})"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("  KONVERSI YOLOv8 → TFLite")
    print("  Target: Raspberry Pi 4 Model B")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.batch:
        convert_batch(args.model_dir, OUTPUT_DIR, args.int8)
    elif args.input:
        result = convert_single_model(args.input, OUTPUT_DIR, args.int8, args.imgsz)
        if result:
            print(f"\n  Langkah selanjutnya:")
            print(f"  1. Transfer file .tflite ke Raspberry Pi 4:")
            print(f"     scp {result} pi@<raspi-ip>:~/benchmark/models/")
            print(f"")
            print(f"  2. Di Raspberry Pi 4, jalankan benchmark:")
            print(f"     python3 benchmark_raspi.py --model {os.path.basename(result)}")
    else:
        # Default: konversi model yolov8n yang sudah ada
        default_model = os.path.join(
            ROOT_DIR,
            "runs_yolov8n_wisard", "runs_yolov8n_wisard",
            "yolov8n_optimized", "weights", "best.pt"
        )
        if os.path.exists(default_model):
            print(f"\n[INFO] Menggunakan model default: {default_model}")
            convert_single_model(default_model, OUTPUT_DIR, args.int8, args.imgsz)
        else:
            print("\n[ERROR] Tidak ada model yang dispesifikasikan!")
            print("        Gunakan: --input path/to/best.pt")
            print("        Atau   : --batch (konversi semua di folder collected)")
            parser.print_help()


if __name__ == "__main__":
    main()
