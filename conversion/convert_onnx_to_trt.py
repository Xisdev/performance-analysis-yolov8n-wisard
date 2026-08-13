"""
==============================================================================
  STEP 2: Konversi ONNX → TensorRT Engine (.engine)
==============================================================================
  *** SCRIPT INI HARUS DIJALANKAN DI NVIDIA JETSON NANO ***
  
  TensorRT engine bersifat hardware-specific, jadi konversi HARUS dilakukan
  di perangkat target (Jetson Nano).
  
  Prasyarat di Jetson Nano:
      - JetPack SDK sudah terinstall (sudah include TensorRT)
      - Python 3.6+
      - pycuda: pip3 install pycuda
  
  Cara pakai:
      python3 convert_onnx_to_trt.py
      
  Opsi tambahan:
      python3 convert_onnx_to_trt.py --fp16          # FP16 (default, recommended)
      python3 convert_onnx_to_trt.py --fp32          # FP32 (lebih akurat, lebih lambat)
      python3 convert_onnx_to_trt.py --int8           # INT8 (butuh kalibrasi)
      python3 convert_onnx_to_trt.py --input best.onnx # Specify input file
==============================================================================
"""

import os
import sys
import time
import argparse


# ============================================================================
#  KONFIGURASI DEFAULT
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONVERT_RESULT_DIR = os.path.join(SCRIPT_DIR, "convert_result")

# Default input ONNX file
DEFAULT_ONNX_FILE = os.path.join(CONVERT_RESULT_DIR, "best.onnx")

# Ukuran input (harus sama dengan saat export ONNX)
IMG_SIZE = 640

# Batch size
BATCH_SIZE = 1

# Max workspace size (dalam MB) - Jetson Nano punya RAM 4GB, set konservatif
MAX_WORKSPACE_MB = 1024  # 1 GB


# ============================================================================
#  METODE 1: Konversi menggunakan Ultralytics (Paling Mudah)
# ============================================================================

def convert_with_ultralytics(onnx_path, output_dir, use_fp16=True):
    """
    Konversi menggunakan Ultralytics YOLO export.
    Metode paling mudah jika ultralytics terinstall di Jetson Nano.
    """
    print("\n[METHOD] Menggunakan Ultralytics YOLO export...")
    
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Ultralytics tidak terinstall.")
        print("        Install: pip3 install ultralytics")
        return None
    
    # Load ONNX model
    model = YOLO(onnx_path, task="detect")
    
    # Export ke TensorRT
    start_time = time.time()
    engine_path = model.export(
        format="engine",
        imgsz=IMG_SIZE,
        half=use_fp16,
        device=0,
        workspace=MAX_WORKSPACE_MB / 1024,  # Ultralytics uses GB
    )
    elapsed = time.time() - start_time
    
    if engine_path and os.path.exists(engine_path):
        # Pindahkan ke output dir
        import shutil
        dest = os.path.join(output_dir, os.path.basename(engine_path))
        if str(engine_path) != dest:
            shutil.move(str(engine_path), dest)
        print(f"\n[OK] Engine berhasil dibuat: {dest}")
        print(f"     Waktu konversi: {elapsed:.1f} detik")
        print(f"     Ukuran: {os.path.getsize(dest) / (1024*1024):.2f} MB")
        return dest
    
    return None


# ============================================================================
#  METODE 2: Konversi menggunakan TensorRT Python API (Lebih Fleksibel)
# ============================================================================

def convert_with_tensorrt_api(onnx_path, output_dir, use_fp16=True, use_int8=False):
    """
    Konversi langsung menggunakan TensorRT Python API.
    Memberikan kontrol penuh atas parameter konversi.
    """
    print("\n[METHOD] Menggunakan TensorRT Python API...")
    
    try:
        import tensorrt as trt
    except ImportError:
        print("[ERROR] TensorRT Python binding tidak ditemukan.")
        print("        Pastikan JetPack SDK sudah terinstall dengan benar.")
        return None
    
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    
    # Nama file output
    precision = "fp16" if use_fp16 else ("int8" if use_int8 else "fp32")
    engine_filename = os.path.splitext(os.path.basename(onnx_path))[0] + f"_{precision}.engine"
    engine_path = os.path.join(output_dir, engine_filename)
    
    print(f"[INFO] Input ONNX    : {onnx_path}")
    print(f"[INFO] Output Engine : {engine_path}")
    print(f"[INFO] Precision     : {precision.upper()}")
    print(f"[INFO] Image Size    : {IMG_SIZE}x{IMG_SIZE}")
    print(f"[INFO] Batch Size    : {BATCH_SIZE}")
    print(f"[INFO] Workspace     : {MAX_WORKSPACE_MB} MB")
    
    # ---- Build Engine ----
    print("\n[1/3] Membuat TensorRT builder dan network...")
    
    EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()
    
    # Set workspace
    config.max_workspace_size = MAX_WORKSPACE_MB * (1024 ** 2)  # Convert to bytes
    
    # Set precision
    if use_fp16:
        if builder.platform_has_fast_fp16:
            print("[INFO] FP16 mode: AKTIF (hardware mendukung)")
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            print("[WARNING] Hardware tidak mendukung FP16, menggunakan FP32")
    
    if use_int8:
        if builder.platform_has_fast_int8:
            print("[INFO] INT8 mode: AKTIF (hardware mendukung)")
            config.set_flag(trt.BuilderFlag.INT8)
            print("[WARNING] INT8 membutuhkan kalibrasi dataset untuk akurasi optimal!")
        else:
            print("[WARNING] Hardware tidak mendukung INT8")
    
    # ---- Parse ONNX ----
    print("\n[2/3] Parsing model ONNX...")
    start_time = time.time()
    
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("[ERROR] Gagal parsing ONNX model!")
            for error_idx in range(parser.num_errors):
                print(f"        Error {error_idx}: {parser.get_error(error_idx)}")
            return None
    
    print(f"       Parsing berhasil!")
    print(f"       - Input layers  : {network.num_inputs}")
    print(f"       - Output layers : {network.num_outputs}")
    
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        print(f"       - Input [{i}] : {input_tensor.name} | shape: {input_tensor.shape} | dtype: {input_tensor.dtype}")
    
    for i in range(network.num_outputs):
        output_tensor = network.get_output(i)
        print(f"       - Output [{i}]: {output_tensor.name} | shape: {output_tensor.shape} | dtype: {output_tensor.dtype}")
    
    # ---- Build Engine ----
    print("\n[3/3] Building TensorRT engine (ini bisa memakan waktu beberapa menit)...")
    print("       Mohon tunggu...")
    
    build_start = time.time()
    
    # TensorRT 8.x+ uses build_serialized_network
    try:
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            print("[ERROR] Gagal build engine!")
            return None
        
        # Simpan engine
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)
    except AttributeError:
        # Fallback untuk TensorRT versi lama (7.x)
        engine = builder.build_engine(network, config)
        if engine is None:
            print("[ERROR] Gagal build engine!")
            return None
        
        # Serialize dan simpan
        with open(engine_path, "wb") as f:
            f.write(engine.serialize())
    
    build_elapsed = time.time() - build_start
    total_elapsed = time.time() - start_time
    
    file_size = os.path.getsize(engine_path) / (1024 * 1024)
    
    print(f"\n[OK] Engine berhasil dibuat!")
    print(f"     File    : {engine_path}")
    print(f"     Ukuran  : {file_size:.2f} MB")
    print(f"     Build   : {build_elapsed:.1f} detik")
    print(f"     Total   : {total_elapsed:.1f} detik")
    
    return engine_path


# ============================================================================
#  METODE 3: Konversi menggunakan trtexec (Command Line Tool)
# ============================================================================

def convert_with_trtexec(onnx_path, output_dir, use_fp16=True):
    """
    Konversi menggunakan trtexec command-line tool.
    Tool ini sudah terinstall bersama TensorRT di JetPack.
    """
    import subprocess
    
    print("\n[METHOD] Menggunakan trtexec command-line tool...")
    
    precision = "fp16" if use_fp16 else "fp32"
    engine_filename = os.path.splitext(os.path.basename(onnx_path))[0] + f"_{precision}.engine"
    engine_path = os.path.join(output_dir, engine_filename)
    
    # Build trtexec command
    cmd = [
        "/usr/src/tensorrt/bin/trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--workspace={MAX_WORKSPACE_MB}",
    ]
    
    if use_fp16:
        cmd.append("--fp16")
    
    # Tambahkan verbose untuk debug
    cmd.append("--verbose")
    
    print(f"[INFO] Command: {' '.join(cmd)}")
    print(f"[INFO] Ini bisa memakan waktu beberapa menit...")
    
    try:
        # Cek apakah trtexec ada
        result = subprocess.run(
            ["/usr/src/tensorrt/bin/trtexec", "--help"],
            capture_output=True, text=True, timeout=10
        )
        
        if result.returncode != 0:
            # Coba path alternatif
            cmd[0] = "trtexec"
            
    except FileNotFoundError:
        # Coba path alternatif
        cmd[0] = "trtexec"
    except subprocess.TimeoutExpired:
        pass
    
    start_time = time.time()
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Stream output
        for line in process.stdout:
            print(f"  [trtexec] {line.rstrip()}")
        
        process.wait()
        elapsed = time.time() - start_time
        
        if process.returncode == 0 and os.path.exists(engine_path):
            file_size = os.path.getsize(engine_path) / (1024 * 1024)
            print(f"\n[OK] Engine berhasil dibuat!")
            print(f"     File   : {engine_path}")
            print(f"     Ukuran : {file_size:.2f} MB")
            print(f"     Waktu  : {elapsed:.1f} detik")
            return engine_path
        else:
            print(f"\n[ERROR] trtexec gagal (return code: {process.returncode})")
            return None
            
    except FileNotFoundError:
        print("[ERROR] trtexec tidak ditemukan!")
        print("        Pastikan TensorRT sudah terinstall via JetPack")
        print("        Biasanya ada di: /usr/src/tensorrt/bin/trtexec")
        return None


# ============================================================================
#  MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Konversi ONNX → TensorRT Engine untuk Jetson Nano"
    )
    parser.add_argument("--input", "-i", type=str, default=DEFAULT_ONNX_FILE,
                        help=f"Path ke file ONNX (default: {DEFAULT_ONNX_FILE})")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Gunakan FP16 precision (default, recommended untuk Jetson Nano)")
    parser.add_argument("--fp32", action="store_true",
                        help="Gunakan FP32 precision (lebih akurat, lebih lambat)")
    parser.add_argument("--int8", action="store_true",
                        help="Gunakan INT8 precision (butuh kalibrasi)")
    parser.add_argument("--method", "-m", type=str, default="auto",
                        choices=["auto", "ultralytics", "tensorrt", "trtexec"],
                        help="Metode konversi (default: auto)")
    
    args = parser.parse_args()
    
    # Determine precision
    use_fp16 = True
    if args.fp32:
        use_fp16 = False
    
    print("=" * 70)
    print("  KONVERSI ONNX → TensorRT Engine")
    print("  Target: NVIDIA Jetson Nano")
    print("=" * 70)
    
    # Validasi input
    onnx_path = args.input
    if not os.path.exists(onnx_path):
        print(f"\n[ERROR] File ONNX tidak ditemukan: {onnx_path}")
        print(f"        Pastikan file sudah ditransfer ke Jetson Nano.")
        print(f"        Atau jalankan convert_to_onnx.py terlebih dahulu.")
        sys.exit(1)
    
    # Buat folder output
    os.makedirs(CONVERT_RESULT_DIR, exist_ok=True)
    
    file_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"\n[INFO] Input file : {onnx_path} ({file_size:.2f} MB)")
    print(f"[INFO] Output dir : {CONVERT_RESULT_DIR}")
    precision = "FP16" if use_fp16 else ("INT8" if args.int8 else "FP32")
    print(f"[INFO] Precision  : {precision}")
    print(f"[INFO] Method     : {args.method}")
    
    engine_path = None
    
    # ---- Auto: coba semua metode ----
    if args.method == "auto":
        print("\n[AUTO] Mencoba metode terbaik yang tersedia...")
        
        # Coba Ultralytics dulu (paling mudah)
        try:
            engine_path = convert_with_ultralytics(onnx_path, CONVERT_RESULT_DIR, use_fp16)
        except Exception as e:
            print(f"[AUTO] Ultralytics gagal: {e}")
        
        # Jika gagal, coba TensorRT API
        if not engine_path:
            try:
                engine_path = convert_with_tensorrt_api(
                    onnx_path, CONVERT_RESULT_DIR, use_fp16, args.int8
                )
            except Exception as e:
                print(f"[AUTO] TensorRT API gagal: {e}")
        
        # Jika masih gagal, coba trtexec
        if not engine_path:
            try:
                engine_path = convert_with_trtexec(onnx_path, CONVERT_RESULT_DIR, use_fp16)
            except Exception as e:
                print(f"[AUTO] trtexec gagal: {e}")
    
    # ---- Metode spesifik ----
    elif args.method == "ultralytics":
        engine_path = convert_with_ultralytics(onnx_path, CONVERT_RESULT_DIR, use_fp16)
    elif args.method == "tensorrt":
        engine_path = convert_with_tensorrt_api(onnx_path, CONVERT_RESULT_DIR, use_fp16, args.int8)
    elif args.method == "trtexec":
        engine_path = convert_with_trtexec(onnx_path, CONVERT_RESULT_DIR, use_fp16)
    
    # ---- Hasil ----
    print("\n" + "=" * 70)
    if engine_path:
        print("  ✓ KONVERSI BERHASIL!")
        print(f"  Engine file: {engine_path}")
        print("\n  Untuk menggunakan engine di Jetson Nano:")
        print("  ─────────────────────────────────────────")
        print("  from ultralytics import YOLO")
        print(f'  model = YOLO("{os.path.basename(engine_path)}")')
        print('  results = model.predict(source="image.jpg")')
    else:
        print("  ✗ KONVERSI GAGAL!")
        print("\n  Troubleshooting:")
        print("  1. Pastikan script ini dijalankan di Jetson Nano")
        print("  2. Pastikan JetPack SDK terinstall")
        print("  3. Install pycuda: pip3 install pycuda")
        print("  4. Install ultralytics: pip3 install ultralytics")
        print("  5. Coba metode lain: --method tensorrt / --method trtexec")
    print("=" * 70)


if __name__ == "__main__":
    main()
