"""
==============================================================================
  YOLOv8 Benchmark Script — NVIDIA Jetson Nano
==============================================================================
  *** SCRIPT INI DIJALANKAN DI NVIDIA JETSON NANO ***

  Mengukur performa inferensi model TensorRT (.engine) dengan metrik:
    - FPS (Frames Per Second)
    - Latency (Preprocessing, Inference, Postprocessing, Total)
    - Resource Usage (CPU, GPU, RAM, Temperature)

  Cara pakai:
      # Benchmark satu model
      python3 benchmark_jetson.py --model best_fp16.engine --images ./test_images/images/

      # Benchmark semua model .engine di folder
      python3 benchmark_jetson.py --model-dir ./models/ --images ./test_images/images/

      # Kustomisasi warm-up dan confidence
      python3 benchmark_jetson.py --model best_fp16.engine --images ./test_images/images/ \\
          --warmup 20 --conf 0.25 --imgsz 640

  Output:
      benchmark_fps_latency_{model}_{device}.csv
      benchmark_resource_{model}_{device}.csv
      benchmark_summary_{model}_{device}.json
==============================================================================
"""

import os
import sys
import csv
import json
import time
import glob
import platform
import argparse
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] Ultralytics belum terinstall!")
    print("        Jalankan: pip3 install ultralytics")
    sys.exit(1)


# ============================================================================
#  KONFIGURASI
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE_NAME = "jetson_nano"

# Default settings
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
DEFAULT_IMGSZ = 640
DEFAULT_WARMUP = 20  # jumlah frame warm-up

# Ekstensi gambar yang didukung
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ============================================================================
#  SYSTEM MONITOR (Background Thread)
# ============================================================================

class SystemMonitor:
    """
    Monitor resource sistem di background thread.
    Khusus untuk Jetson Nano: menggunakan tegrastats dan file system.
    """

    def __init__(self, csv_path, interval=1.0):
        self.csv_path = csv_path
        self.interval = interval
        self._running = False
        self._thread = None
        self._start_time = None

    def start(self):
        """Mulai monitoring di background thread."""
        self._running = True
        self._start_time = time.time()

        # Tulis header CSV
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'elapsed_s',
                'cpu_percent', 'gpu_percent',
                'ram_used_mb', 'ram_total_mb',
                'temperature_c'
            ])

        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Hentikan monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def _monitor_loop(self):
        """Loop monitoring."""
        while self._running:
            try:
                elapsed = time.time() - self._start_time
                cpu = self._get_cpu_usage()
                gpu = self._get_gpu_usage()
                ram_used, ram_total = self._get_ram_usage()
                temp = self._get_temperature()

                with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now().isoformat(),
                        f"{elapsed:.1f}",
                        f"{cpu:.1f}",
                        f"{gpu:.1f}",
                        f"{ram_used:.0f}",
                        f"{ram_total:.0f}",
                        f"{temp:.1f}"
                    ])
            except Exception:
                pass

            time.sleep(self.interval)

    def _get_cpu_usage(self):
        """Baca CPU usage dari /proc/stat."""
        try:
            import psutil
            return psutil.cpu_percent(interval=None)
        except ImportError:
            pass

        try:
            with open('/proc/loadavg', 'r') as f:
                load = float(f.read().split()[0])
                # Normalisasi ke persen (4 core)
                return min(load / 4.0 * 100, 100.0)
        except Exception:
            return 0.0

    def _get_gpu_usage(self):
        """Baca GPU usage Jetson Nano dari sysfs."""
        # Jetson Nano: GPU load dari sysfs
        gpu_load_paths = [
            '/sys/devices/gpu.0/load',
            '/sys/devices/57000000.gpu/load',
            '/sys/devices/platform/gpu.0/load',
        ]
        for path in gpu_load_paths:
            try:
                with open(path, 'r') as f:
                    # Nilai dalam per-mille (0-1000)
                    return float(f.read().strip()) / 10.0
            except (FileNotFoundError, ValueError):
                continue

        # Fallback: coba tegrastats
        try:
            result = subprocess.run(
                ['tegrastats', '--interval', '100', '--count', '1'],
                capture_output=True, text=True, timeout=2
            )
            if 'GR3D' in result.stdout:
                # Parse "GR3D_FREQ 76%"
                for part in result.stdout.split():
                    if '%' in part and part.replace('%', '').isdigit():
                        return float(part.replace('%', ''))
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return 0.0

    def _get_ram_usage(self):
        """Baca RAM usage."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return mem.used / (1024 * 1024), mem.total / (1024 * 1024)
        except ImportError:
            pass

        try:
            with open('/proc/meminfo', 'r') as f:
                lines = f.readlines()
                info = {}
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0].rstrip(':')
                        val = int(parts[1])  # dalam kB
                        info[key] = val

                total = info.get('MemTotal', 0) / 1024  # MB
                available = info.get('MemAvailable', 0) / 1024
                used = total - available
                return used, total
        except Exception:
            return 0.0, 0.0

    def _get_temperature(self):
        """Baca suhu CPU/GPU Jetson Nano."""
        # Jetson Nano thermal zones
        thermal_paths = [
            '/sys/devices/virtual/thermal/thermal_zone0/temp',
            '/sys/devices/virtual/thermal/thermal_zone1/temp',
            '/sys/class/thermal/thermal_zone0/temp',
        ]
        temps = []
        for path in thermal_paths:
            try:
                with open(path, 'r') as f:
                    temp = float(f.read().strip()) / 1000.0  # Konversi mili-Celsius
                    temps.append(temp)
            except (FileNotFoundError, ValueError):
                continue

        return max(temps) if temps else 0.0


# ============================================================================
#  BENCHMARK ENGINE
# ============================================================================

def get_image_files(image_dir):
    """Ambil semua file gambar dari direktori."""
    files = []
    for f in sorted(os.listdir(image_dir)):
        if Path(f).suffix.lower() in IMG_EXTENSIONS:
            files.append(f)
    return files


def run_benchmark(model_path, image_dir, output_dir,
                  conf=DEFAULT_CONF, iou=DEFAULT_IOU,
                  imgsz=DEFAULT_IMGSZ, warmup=DEFAULT_WARMUP,
                  device="0"):
    """
    Jalankan benchmark untuk satu model.

    Pengukuran timestamp:
        T0: Sebelum preprocessing (baca gambar + resize)
        T1: Setelah preprocessing, sebelum inferensi model
        T2: Setelah inferensi model, sebelum postprocessing (NMS)
        T3: Setelah postprocessing (NMS + filter)

    Karena Ultralytics menggabungkan pre/infer/post dalam satu panggilan,
    kita menggunakan `result.speed` yang secara internal mengukur timing
    dari ketiga tahap tersebut.
    """
    model_name = Path(model_path).stem
    print(f"\n{'=' * 70}")
    print(f"  BENCHMARK: {model_name}")
    print(f"  Device   : {DEVICE_NAME}")
    print(f"{'=' * 70}")

    # Validasi
    if not os.path.exists(model_path):
        print(f"[ERROR] Model tidak ditemukan: {model_path}")
        return None

    if not os.path.isdir(image_dir):
        print(f"[ERROR] Folder gambar tidak ditemukan: {image_dir}")
        return None

    image_files = get_image_files(image_dir)
    if not image_files:
        print(f"[ERROR] Tidak ada gambar ditemukan di: {image_dir}")
        return None

    print(f"  Model    : {model_path}")
    print(f"  Images   : {image_dir} ({len(image_files)} gambar)")
    print(f"  Config   : imgsz={imgsz}, conf={conf}, iou={iou}")
    print(f"  Warm-up  : {warmup} frame")
    print(f"  Device   : {device}")

    # Load model
    print(f"\n[1/4] Loading model...")
    t_load_start = time.time()

    try:
        model = YOLO(model_path, task="detect")
    except Exception as e:
        print(f"[ERROR] Gagal load model: {e}")
        return None

    t_load = time.time() - t_load_start
    print(f"       Model loaded dalam {t_load:.1f}s")

    # Info model
    model_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"       Ukuran model: {model_size:.2f} MB")

    # Setup output CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_fps_path = os.path.join(output_dir, f"benchmark_fps_latency_{model_name}_{DEVICE_NAME}.csv")
    csv_resource_path = os.path.join(output_dir, f"benchmark_resource_{model_name}_{DEVICE_NAME}.csv")
    json_summary_path = os.path.join(output_dir, f"benchmark_summary_{model_name}_{DEVICE_NAME}.json")

    # Start system monitor
    print(f"\n[2/4] Starting system monitor...")
    monitor = SystemMonitor(csv_resource_path, interval=1.0)
    monitor.start()

    # Warm-up
    print(f"\n[3/4] Warm-up ({warmup} frames)...")
    warmup_images = image_files[:warmup] if len(image_files) >= warmup else image_files
    for i, img_file in enumerate(warmup_images):
        img_path = os.path.join(image_dir, img_file)
        try:
            model.predict(
                source=img_path,
                conf=conf, iou=iou, imgsz=imgsz,
                device=device, verbose=False,
            )
        except Exception as e:
            print(f"       [WARN] Warm-up frame {i} gagal: {e}")

        if (i + 1) % 5 == 0:
            print(f"       Warm-up {i + 1}/{len(warmup_images)}")

    print(f"       Warm-up selesai.")

    # Benchmark utama
    print(f"\n[4/4] Running benchmark ({len(image_files)} images)...")

    # Tulis header CSV
    with open(csv_fps_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'frame_idx', 'image_file',
            'preprocess_ms', 'inference_ms', 'postprocess_ms', 'total_ms',
            'fps', 'num_detections', 'is_warmup'
        ])

    all_preprocess = []
    all_inference = []
    all_postprocess = []
    all_total = []
    all_fps = []
    all_detections = []

    t_benchmark_start = time.time()

    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(image_dir, img_file)

        try:
            # --- TIMING: Total end-to-end ---
            t0 = time.perf_counter()

            results = model.predict(
                source=img_path,
                conf=conf, iou=iou, imgsz=imgsz,
                device=device, verbose=False,
            )

            t3 = time.perf_counter()

            result = results[0]

            # --- Ambil timing internal dari Ultralytics ---
            # result.speed berisi: {'preprocess': ms, 'inference': ms, 'postprocess': ms}
            speed = result.speed
            preprocess_ms = speed.get('preprocess', 0.0)
            inference_ms = speed.get('inference', 0.0)
            postprocess_ms = speed.get('postprocess', 0.0)

            # Total dari Ultralytics
            total_internal = preprocess_ms + inference_ms + postprocess_ms

            # Total dari pengukuran eksternal
            total_external = (t3 - t0) * 1000

            # Gunakan yang lebih akurat (internal dari Ultralytics)
            total_ms = total_internal
            fps = 1000.0 / total_ms if total_ms > 0 else 0.0

            # Jumlah deteksi
            num_det = len(result.boxes) if result.boxes is not None else 0

            # Simpan ke list (hanya non-warmup, tapi record semua di CSV)
            is_warmup = idx < warmup

            if not is_warmup:
                all_preprocess.append(preprocess_ms)
                all_inference.append(inference_ms)
                all_postprocess.append(postprocess_ms)
                all_total.append(total_ms)
                all_fps.append(fps)
                all_detections.append(num_det)

            # Tulis ke CSV
            with open(csv_fps_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    idx + 1, img_file,
                    f"{preprocess_ms:.2f}", f"{inference_ms:.2f}",
                    f"{postprocess_ms:.2f}", f"{total_ms:.2f}",
                    f"{fps:.2f}", num_det,
                    "true" if is_warmup else "false"
                ])

        except Exception as e:
            print(f"  [ERROR] Frame {idx + 1} ({img_file}): {e}")
            # Tulis error row
            with open(csv_fps_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    idx + 1, img_file,
                    "ERROR", "ERROR", "ERROR", "ERROR",
                    "0", "0", "true" if idx < warmup else "false"
                ])
            continue

        # Progress
        if (idx + 1) % 50 == 0 or (idx + 1) == len(image_files):
            avg_fps = np.mean(all_fps) if all_fps else 0
            print(f"  Frame {idx + 1:5d}/{len(image_files)} | "
                  f"Infer: {inference_ms:6.1f}ms | "
                  f"Total: {total_ms:6.1f}ms | "
                  f"FPS: {fps:5.1f} | "
                  f"Avg FPS: {avg_fps:5.1f} | "
                  f"Det: {num_det}")

    t_benchmark_total = time.time() - t_benchmark_start

    # Stop monitor
    monitor.stop()

    # Hitung statistik
    if not all_total:
        print("[ERROR] Tidak ada frame yang berhasil diproses!")
        return None

    stats = {
        'model_name': model_name,
        'model_path': model_path,
        'model_size_mb': round(model_size, 2),
        'device': DEVICE_NAME,
        'image_count': len(image_files),
        'warmup_count': warmup,
        'benchmark_count': len(all_total),
        'config': {
            'imgsz': imgsz,
            'conf': conf,
            'iou': iou,
        },
        'latency': {
            'preprocess_ms': {
                'mean': round(np.mean(all_preprocess), 2),
                'median': round(np.median(all_preprocess), 2),
                'std': round(np.std(all_preprocess), 2),
                'min': round(np.min(all_preprocess), 2),
                'max': round(np.max(all_preprocess), 2),
                'p95': round(np.percentile(all_preprocess, 95), 2),
                'p99': round(np.percentile(all_preprocess, 99), 2),
            },
            'inference_ms': {
                'mean': round(np.mean(all_inference), 2),
                'median': round(np.median(all_inference), 2),
                'std': round(np.std(all_inference), 2),
                'min': round(np.min(all_inference), 2),
                'max': round(np.max(all_inference), 2),
                'p95': round(np.percentile(all_inference, 95), 2),
                'p99': round(np.percentile(all_inference, 99), 2),
            },
            'postprocess_ms': {
                'mean': round(np.mean(all_postprocess), 2),
                'median': round(np.median(all_postprocess), 2),
                'std': round(np.std(all_postprocess), 2),
                'min': round(np.min(all_postprocess), 2),
                'max': round(np.max(all_postprocess), 2),
                'p95': round(np.percentile(all_postprocess, 95), 2),
                'p99': round(np.percentile(all_postprocess, 99), 2),
            },
            'total_ms': {
                'mean': round(np.mean(all_total), 2),
                'median': round(np.median(all_total), 2),
                'std': round(np.std(all_total), 2),
                'min': round(np.min(all_total), 2),
                'max': round(np.max(all_total), 2),
                'p95': round(np.percentile(all_total, 95), 2),
                'p99': round(np.percentile(all_total, 99), 2),
            },
        },
        'fps': {
            'mean': round(np.mean(all_fps), 2),
            'median': round(np.median(all_fps), 2),
            'std': round(np.std(all_fps), 2),
            'min': round(np.min(all_fps), 2),
            'max': round(np.max(all_fps), 2),
            'p5': round(np.percentile(all_fps, 5), 2),
        },
        'detections': {
            'total': int(sum(all_detections)),
            'mean_per_frame': round(np.mean(all_detections), 2),
        },
        'timing': {
            'model_load_s': round(t_load, 2),
            'benchmark_total_s': round(t_benchmark_total, 2),
        },
        'system': {
            'hostname': platform.node(),
            'platform': platform.platform(),
            'python': platform.python_version(),
            'timestamp': datetime.now().isoformat(),
        },
    }

    # Simpan summary JSON
    with open(json_summary_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"  HASIL BENCHMARK: {model_name}")
    print(f"{'=' * 70}")
    print(f"  Device              : {DEVICE_NAME}")
    print(f"  Model size          : {model_size:.2f} MB")
    print(f"  Frame diproses      : {stats['benchmark_count']} (excl. {warmup} warmup)")
    print(f"")
    print(f"  ── Latency (ms) ────────────────────────────")
    print(f"  Preprocess  : {stats['latency']['preprocess_ms']['mean']:7.2f} ± {stats['latency']['preprocess_ms']['std']:.2f}")
    print(f"  Inference   : {stats['latency']['inference_ms']['mean']:7.2f} ± {stats['latency']['inference_ms']['std']:.2f}")
    print(f"  Postprocess : {stats['latency']['postprocess_ms']['mean']:7.2f} ± {stats['latency']['postprocess_ms']['std']:.2f}")
    print(f"  Total       : {stats['latency']['total_ms']['mean']:7.2f} ± {stats['latency']['total_ms']['std']:.2f}")
    print(f"")
    print(f"  ── FPS ─────────────────────────────────────")
    print(f"  Mean        : {stats['fps']['mean']:7.2f}")
    print(f"  Median      : {stats['fps']['median']:7.2f}")
    print(f"  Min         : {stats['fps']['min']:7.2f}")
    print(f"  Max         : {stats['fps']['max']:7.2f}")
    print(f"")
    print(f"  ── Output Files ────────────────────────────")
    print(f"  📄 {csv_fps_path}")
    print(f"  📄 {csv_resource_path}")
    print(f"  📄 {json_summary_path}")
    print(f"{'=' * 70}")

    return stats


# ============================================================================
#  MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark YOLOv8 TensorRT di NVIDIA Jetson Nano"
    )
    parser.add_argument(
        "--model", "-m", type=str, default=None,
        help="Path ke file .engine (model TensorRT)"
    )
    parser.add_argument(
        "--model-dir", type=str, default=None,
        help="Folder berisi file .engine (benchmark semua model)"
    )
    parser.add_argument(
        "--images", "-i", type=str, required=True,
        help="Path ke folder gambar test"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=SCRIPT_DIR,
        help=f"Folder output CSV/JSON (default: {SCRIPT_DIR})"
    )
    parser.add_argument(
        "--conf", type=float, default=DEFAULT_CONF,
        help=f"Confidence threshold (default: {DEFAULT_CONF})"
    )
    parser.add_argument(
        "--iou", type=float, default=DEFAULT_IOU,
        help=f"IoU threshold NMS (default: {DEFAULT_IOU})"
    )
    parser.add_argument(
        "--imgsz", type=int, default=DEFAULT_IMGSZ,
        help=f"Ukuran input model (default: {DEFAULT_IMGSZ})"
    )
    parser.add_argument(
        "--warmup", "-w", type=int, default=DEFAULT_WARMUP,
        help=f"Jumlah frame warm-up (default: {DEFAULT_WARMUP})"
    )
    parser.add_argument(
        "--device", "-d", type=str, default="0",
        help="Device: '0' (GPU, default), 'cpu'"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("  YOLOv8 BENCHMARK — NVIDIA Jetson Nano")
    print("  Metrik: FPS, Latency, Resource Usage")
    print("=" * 70)

    all_results = []

    if args.model_dir:
        # Benchmark semua model di folder
        engine_files = sorted(glob.glob(os.path.join(args.model_dir, "*.engine")))
        if not engine_files:
            print(f"[ERROR] Tidak ada file .engine di: {args.model_dir}")
            sys.exit(1)

        print(f"\n[BATCH] Ditemukan {len(engine_files)} model:")
        for f in engine_files:
            print(f"        - {os.path.basename(f)}")

        for engine_file in engine_files:
            result = run_benchmark(
                model_path=engine_file,
                image_dir=args.images,
                output_dir=args.output,
                conf=args.conf, iou=args.iou,
                imgsz=args.imgsz, warmup=args.warmup,
                device=args.device,
            )
            if result:
                all_results.append(result)

    elif args.model:
        # Benchmark satu model
        result = run_benchmark(
            model_path=args.model,
            image_dir=args.images,
            output_dir=args.output,
            conf=args.conf, iou=args.iou,
            imgsz=args.imgsz, warmup=args.warmup,
            device=args.device,
        )
        if result:
            all_results.append(result)

    else:
        print("[ERROR] Spesifikasikan model: --model path/to/model.engine")
        print("        Atau benchmark batch: --model-dir path/to/folder/")
        parser.print_help()
        sys.exit(1)

    # Print summary tabel jika batch
    if len(all_results) > 1:
        print(f"\n{'=' * 80}")
        print(f"  RINGKASAN BATCH BENCHMARK — {DEVICE_NAME}")
        print(f"{'=' * 80}")
        print(f"  {'Model':<30s} {'Size(MB)':>10s} {'FPS':>8s} {'Latency(ms)':>12s} {'Infer(ms)':>10s}")
        print(f"  {'─' * 30} {'─' * 10} {'─' * 8} {'─' * 12} {'─' * 10}")
        for r in all_results:
            print(f"  {r['model_name']:<30s} "
                  f"{r['model_size_mb']:>10.2f} "
                  f"{r['fps']['mean']:>8.2f} "
                  f"{r['latency']['total_ms']['mean']:>12.2f} "
                  f"{r['latency']['inference_ms']['mean']:>10.2f}")
        print(f"{'=' * 80}")

    print("\n[DONE] Benchmark selesai!")


if __name__ == "__main__":
    main()
