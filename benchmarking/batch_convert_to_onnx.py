"""
==============================================================================
  BATCH Konversi YOLOv8 .pt -> .onnx (untuk Jetson Nano)
==============================================================================
  Script ini mengkonversi SEMUA model .pt di folder kumpulan_model_ready/
  ke format ONNX dalam sekali jalan.

  Hasil disimpan di: test_for_device/jetson_nano/onnx_model/
  Nama file: best_yolov8n.onnx, best_yolov8s.onnx, dst.

  Jalur konversi: .pt --> .onnx (di PC) --> .engine (di Jetson Nano)

  Cara pakai:
      python "02. batch_convert_to_onnx.py"
  
  Prasyarat:
      pip install ultralytics onnx onnxruntime
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

# Folder output: ONNX untuk Jetson Nano
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'jetson_nano', 'onnx_model')

# Ukuran input (harus sama dengan training)
IMG_SIZE = 640

# Opset ONNX (11 kompatibel dengan Jetson Nano TensorRT)
ONNX_OPSET = 11

# Simplify ONNX (optimisasi graph)
SIMPLIFY = True


# ============================================================================
#  PROSES KONVERSI
# ============================================================================

def convert_one(model_path, output_dir):
    """Konversi satu model .pt ke .onnx."""
    model_name = Path(model_path).stem  # e.g. best_yolov8n
    
    print(f"\n{'-' * 65}")
    print(f"  [{model_name}]")
    print(f"{'-' * 65}")
    print(f"  Input  : {model_path}")
    print(f"  Output : {output_dir}/{model_name}.onnx")
    
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
    
    # Export ke ONNX
    print(f"  [2/3] Exporting to ONNX (opset={ONNX_OPSET}, simplify={SIMPLIFY})...")
    start_time = time.time()
    
    try:
        export_path = model.export(
            format="onnx",
            imgsz=IMG_SIZE,
            opset=ONNX_OPSET,
            simplify=SIMPLIFY,
            dynamic=False,      # Static shape untuk Jetson Nano
            half=False,         # FP32 ONNX, FP16 dilakukan saat konversi TensorRT
        )
    except Exception as e:
        print(f"  [ERROR] Gagal export: {e}")
        return None
    
    elapsed = time.time() - start_time
    print(f"         Selesai dalam {elapsed:.1f} detik")
    
    # Pindahkan file ONNX ke folder output
    print(f"  [3/3] Memindahkan ke output folder...")
    
    # Cari file ONNX yang dihasilkan
    onnx_path = None
    if export_path and os.path.exists(str(export_path)):
        onnx_path = str(export_path)
    else:
        # Cari di sekitar model path
        candidates = [
            model_path.replace(".pt", ".onnx"),
            os.path.join(os.path.dirname(model_path), model_name + ".onnx"),
        ]
        for c in candidates:
            if os.path.exists(c):
                onnx_path = c
                break
    
    if onnx_path and os.path.exists(onnx_path):
        dest_path = os.path.join(output_dir, f"{model_name}.onnx")
        shutil.move(onnx_path, dest_path)
        
        file_size = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"         File  : {dest_path}")
        print(f"         Size  : {file_size:.1f} MB")
        return dest_path
    else:
        print(f"  [ERROR] File ONNX tidak ditemukan setelah export!")
        return None


def main():
    print("=" * 65)
    print("  BATCH KONVERSI YOLOv8 .pt -> .onnx")
    print("  Target: NVIDIA Jetson Nano (TensorRT)")
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
    
    print(f"\n  Model dir : {MODEL_DIR}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  ONNX Opset: {ONNX_OPSET}")
    print(f"  Simplify  : {SIMPLIFY}")
    print(f"\n  Ditemukan {len(pt_files)} model:")
    for f in pt_files:
        size = os.path.getsize(f) / (1024 * 1024)
        print(f"    - {os.path.basename(f)} ({size:.1f} MB)")
    
    # Konversi satu per satu
    results = []
    total_start = time.time()
    
    for pt_file in pt_files:
        result = convert_one(pt_file, OUTPUT_DIR)
        results.append((os.path.basename(pt_file), result))
    
    total_elapsed = time.time() - total_start
    
    # Ringkasan
    print(f"\n{'=' * 65}")
    print(f"  RINGKASAN KONVERSI .pt -> .onnx")
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
    print(f"  Waktu: {total_elapsed:.1f} detik")
    print(f"  Output: {OUTPUT_DIR}")
    
    if success > 0:
        print(f"\n  Langkah selanjutnya:")
        print(f"  1. Transfer file .onnx ke Jetson Nano:")
        print(f"     scp -r {OUTPUT_DIR}/ <user>@<jetson-ip>:~/benchmark/onnx/")
        print(f"  2. Di Jetson Nano, konversi ke TensorRT:")
        print(f"     python3 convert_onnx_to_trt.py --input onnx/<model>.onnx --fp16")
    
    print(f"\n{'=' * 65}")


if __name__ == "__main__":
    main()
