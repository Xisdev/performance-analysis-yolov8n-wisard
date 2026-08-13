"""
==============================================================================
  BATCH Konversi YOLOv8 .pt -> .tflite (untuk Raspberry Pi 4)
==============================================================================
  Script ini mengkonversi SEMUA model .pt di folder kumpulan_model_ready/
  ke format TensorFlow Lite dalam sekali jalan.

  Hasil disimpan di: test_for_device/raspi4/tflite_model/
  Nama file: best_yolov8n_fp32.tflite, best_yolov8s_fp32.tflite, dst.

  Cara pakai:
      python "03. batch_convert_to_tflite.py"

  Prasyarat:
      pip install ultralytics tensorflow
==============================================================================
"""

import os
import sys
import time
import shutil
import glob
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

# Folder input: model .pt yang sudah di-strip
MODEL_DIR = os.path.join(SCRIPT_DIR, 'kumpulan_model_ready')

# Folder output: TFLite untuk Raspberry Pi 4
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'raspi4', 'tflite_model')

# Ukuran input (harus sama dengan training)
IMG_SIZE = 640

# Presisi: FP32 (default untuk RPi4 CPU), bisa juga INT8
USE_INT8 = False


# ============================================================================
#  PROSES KONVERSI
# ============================================================================

def convert_one(model_path, output_dir, use_int8=False):
    """Konversi satu model .pt ke .tflite."""
    model_name = Path(model_path).stem  # e.g. best_yolov8n
    precision = "int8" if use_int8 else "fp32"

    print(f"\n{'-' * 65}")
    print(f"  [{model_name}] -> .tflite ({precision.upper()})")
    print(f"{'-' * 65}")
    print(f"  Input  : {model_path}")

    # Ukuran file input
    src_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"  Size   : {src_size:.1f} MB")

    # Load model
    print(f"  [1/3] Loading model...")
    try:
        model = YOLO(model_path)
        print(f"         Task  : {model.task}")
        print(f"         Names : {model.names}")
    except Exception as e:
        print(f"  [ERROR] Gagal load model: {e}")
        return None

    # Export ke TFLite
    print(f"  [2/3] Exporting to TFLite ({precision.upper()})...")
    print(f"         Ini bisa memakan waktu 1-5 menit per model...")
    start_time = time.time()

    try:
        export_path = model.export(
            format="tflite",
            imgsz=IMG_SIZE,
            int8=use_int8,
            half=False,         # TFLite tidak mendukung FP16 di CPU
        )
    except Exception as e:
        print(f"  [ERROR] Gagal export: {e}")
        print(f"  [TIP]  Pastikan tensorflow terinstall: pip install tensorflow")
        return None

    elapsed = time.time() - start_time
    print(f"         Selesai dalam {elapsed:.1f} detik")

    # Cari dan pindahkan file TFLite
    print(f"  [3/3] Memindahkan ke output folder...")
    os.makedirs(output_dir, exist_ok=True)

    tflite_path = None

    # Ultralytics export biasanya return path ke saved_model folder
    if export_path:
        export_str = str(export_path)
        # Cek apakah export_path langsung .tflite
        if export_str.endswith('.tflite') and os.path.exists(export_str):
            tflite_path = export_str
        # Cek di dalam saved_model folder
        elif os.path.isdir(export_str):
            for root, dirs, files in os.walk(export_str):
                for f in files:
                    if f.endswith('.tflite'):
                        tflite_path = os.path.join(root, f)
                        break
                if tflite_path:
                    break

    # Fallback: cari di sekitar model path
    if tflite_path is None:
        search_dir = os.path.dirname(model_path)
        for root, dirs, files in os.walk(search_dir):
            for f in files:
                if f.endswith('.tflite') and model_name in root:
                    tflite_path = os.path.join(root, f)
                    break
            if tflite_path:
                break

    if tflite_path and os.path.exists(tflite_path):
        # Nama output: best_yolov8n_fp32.tflite
        dest_name = f"{model_name}_{precision}.tflite"
        dest_path = os.path.join(output_dir, dest_name)
        shutil.copy2(tflite_path, dest_path)

        file_size = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"         File  : {dest_path}")
        print(f"         Size  : {file_size:.1f} MB")

        # Cleanup saved_model folder (sangat besar, ~300-800MB per model)
        saved_model_dir = model_path.replace(".pt", "_saved_model")
        if os.path.isdir(saved_model_dir):
            print(f"         Cleanup: Menghapus _saved_model/ ({Path(saved_model_dir).name})...")
            shutil.rmtree(saved_model_dir, ignore_errors=True)

        # Cleanup juga di folder lain yang mungkin dibuat Ultralytics
        parent = os.path.dirname(model_path)
        for item in os.listdir(parent):
            item_path = os.path.join(parent, item)
            if os.path.isdir(item_path) and item.endswith('_saved_model'):
                print(f"         Cleanup: {item}/")
                shutil.rmtree(item_path, ignore_errors=True)

        return dest_path
    else:
        print(f"  [ERROR] File TFLite tidak ditemukan setelah export!")
        print(f"  [TIP]  Cek manual di folder model untuk file .tflite")
        return None


def main():
    print("=" * 65)
    print("  BATCH KONVERSI YOLOv8 .pt -> .tflite")
    print("  Target: Raspberry Pi 4 Model B (CPU)")
    print("=" * 65)

    # Validasi input folder
    if not os.path.exists(MODEL_DIR):
        print(f"\n[ERROR] Folder model tidak ditemukan: {MODEL_DIR}")
        print(f"        Jalankan strip_optimizer.py terlebih dahulu!")
        sys.exit(1)

    # Cari semua .pt
    pt_files = sorted(glob.glob(os.path.join(MODEL_DIR, "*.pt")))
    if not pt_files:
        print(f"\n[ERROR] Tidak ada file .pt di: {MODEL_DIR}")
        sys.exit(1)

    # Buat output folder
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    precision = "INT8" if USE_INT8 else "FP32"
    print(f"\n  Model dir : {MODEL_DIR}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Precision : {precision}")
    print(f"\n  Ditemukan {len(pt_files)} model:")
    for f in pt_files:
        size = os.path.getsize(f) / (1024 * 1024)
        print(f"    - {os.path.basename(f)} ({size:.1f} MB)")

    print(f"\n  PERINGATAN: Konversi TFLite memakan waktu cukup lama!")
    print(f"  Estimasi total: 5-25 menit untuk {len(pt_files)} model.")

    # Konversi satu per satu
    results = []
    total_start = time.time()

    for pt_file in pt_files:
        result = convert_one(pt_file, OUTPUT_DIR, USE_INT8)
        results.append((os.path.basename(pt_file), result))

    total_elapsed = time.time() - total_start

    # Ringkasan
    print(f"\n{'=' * 65}")
    print(f"  RINGKASAN KONVERSI .pt -> .tflite")
    print(f"{'=' * 65}")

    success = 0
    for name, path in results:
        if path:
            size = os.path.getsize(path) / (1024 * 1024)
            print(f"  [OK] {name} -> {os.path.basename(path)} ({size:.1f} MB)")
            success += 1
        else:
            print(f"  [!!] {name} -> GAGAL")

    print(f"\n  Total: {success}/{len(results)} berhasil")
    print(f"  Waktu: {total_elapsed:.0f} detik ({total_elapsed/60:.1f} menit)")
    print(f"  Output: {OUTPUT_DIR}")

    if success > 0:
        print(f"\n  Langkah selanjutnya:")
        print(f"  1. Transfer file .tflite ke Raspberry Pi 4:")
        print(f"     scp -r {OUTPUT_DIR}/ pi@<raspi-ip>:~/benchmark/models/")
        print(f"  2. Di Raspberry Pi 4, jalankan benchmark:")
        print(f"     python3 benchmark_raspi.py --model models/<model>.tflite")

    print(f"\n{'=' * 65}")


if __name__ == "__main__":
    main()
