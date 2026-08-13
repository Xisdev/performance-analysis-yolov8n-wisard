"""
==============================================================================
  STEP 3: Inferensi dengan TensorRT Engine (.engine)
==============================================================================
  *** SCRIPT INI DIJALANKAN DI NVIDIA JETSON NANO ***

  Prasyarat:
      pip3 install ultralytics opencv-python

  Cara pakai:
      # Dari gambar
      python3 run_inference.py --source gambar.jpg

      # Dari kamera (real-time)
      python3 run_inference.py --source 0

      # Paksa CPU (jika CUDA belum tersedia)
      python3 run_inference.py --source 0 --device cpu

      # Dari video
      python3 run_inference.py --source video.mp4

      # Ubah confidence threshold
      python3 run_inference.py --source 0 --conf 0.5

      # Simpan hasil video
      python3 run_inference.py --source video.mp4 --save
==============================================================================
"""

import os
import sys
import argparse
import time
import cv2
import torch

# ============================================================================
#  KONFIGURASI
# ============================================================================

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
CONVERT_DIR  = os.path.join(SCRIPT_DIR, "convert_result")

# Cari file .engine otomatis di folder convert_result
def find_engine_file():
    if not os.path.exists(CONVERT_DIR):
        return None
    for f in os.listdir(CONVERT_DIR):
        if f.endswith(".engine"):
            return os.path.join(CONVERT_DIR, f)
    return None

DEFAULT_ENGINE = find_engine_file() or os.path.join(CONVERT_DIR, "best_fp16.engine")
DEFAULT_CONF   = 0.25
DEFAULT_IOU    = 0.45
IMG_SIZE       = 640


def get_device(requested="auto"):
    """Pilih device: auto-detect CUDA, fallback ke CPU dengan peringatan."""
    if requested != "auto":
        return requested

    if torch.cuda.is_available():
        print(f"[DEVICE] CUDA tersedia → menggunakan GPU (cuda:0)")
        print(f"         GPU: {torch.cuda.get_device_name(0)}")
        return "0"
    else:
        print("[WARNING] CUDA tidak tersedia — torch.cuda.is_available() = False")
        print("          Kemungkinan PyTorch yang terinstall adalah versi CPU biasa.")
        print("          TensorRT engine membutuhkan CUDA. Lihat README untuk solusi.")
        print("          Mencoba jalankan dengan CPU (mungkin error untuk .engine)...\n")
        return "cpu"


# ============================================================================
#  LOAD MODEL
# ============================================================================

def load_model(engine_path, device="0"):
    """Load TensorRT engine menggunakan Ultralytics YOLO."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Ultralytics tidak terinstall.")
        print("        Jalankan: pip3 install ultralytics")
        sys.exit(1)

    if not os.path.exists(engine_path):
        print(f"[ERROR] Engine file tidak ditemukan: {engine_path}")
        print(f"        Pastikan konversi sudah dilakukan dengan convert_onnx_to_trt.py")
        sys.exit(1)

    print(f"[INFO] Loading engine: {engine_path}")
    print(f"[INFO] Ukuran file  : {os.path.getsize(engine_path) / (1024*1024):.2f} MB")
    print(f"[INFO] Device       : {device}")

    model = YOLO(engine_path, task="detect")
    print(f"[OK]   Model berhasil dimuat!\n")
    return model


# ============================================================================
#  INFERENSI GAMBAR / VIDEO / KAMERA
# ============================================================================

def run_on_image(model, source, conf, iou, save, device="0"):
    """Jalankan inferensi pada gambar tunggal."""
    print(f"[MODE] Gambar: {source}")

    results = model.predict(
        source=source,
        conf=conf,
        iou=iou,
        imgsz=IMG_SIZE,
        device=device,
        verbose=False,
    )

    result = results[0]

    # Tampilkan deteksi
    boxes  = result.boxes
    names  = result.names
    print(f"\n[RESULT] Terdeteksi {len(boxes)} objek:")
    for box in boxes:
        cls_id = int(box.cls[0])
        conf_v = float(box.conf[0])
        xyxy   = box.xyxy[0].tolist()
        print(f"  - {names[cls_id]:15s}  conf={conf_v:.2f}  bbox={[int(v) for v in xyxy]}")

    # Tampilkan gambar dengan bounding box
    annotated = result.plot()
    cv2.imshow("TensorRT Inference - Jetson Nano", annotated)

    if save:
        out_path = "result_" + os.path.basename(source)
        cv2.imwrite(out_path, annotated)
        print(f"\n[SAVED] Hasil disimpan: {out_path}")

    print("\nTekan tombol apapun untuk keluar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_on_video_or_camera(model, source, conf, iou, save, device="0"):
    """Jalankan inferensi real-time dari kamera atau file video."""
    is_camera = isinstance(source, int) or str(source).isdigit()
    src_int   = int(source) if str(source).isdigit() else source

    if is_camera:
        print(f"[MODE] Kamera (index: {src_int})")
        print("[INFO] Tekan 'q' untuk keluar, 's' untuk screenshot")
    else:
        print(f"[MODE] Video: {source}")
        print("[INFO] Tekan 'q' untuk keluar")

    cap = cv2.VideoCapture(src_int)

    if not cap.isOpened():
        print(f"[ERROR] Tidak bisa membuka sumber: {source}")
        sys.exit(1)

    # Info kamera/video
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] Resolusi: {width}x{height} @ {fps_src:.1f} FPS")

    # Setup video writer jika --save
    writer = None
    if save:
        out_name = "result_camera.mp4" if is_camera else "result_" + os.path.basename(str(source))
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(out_name, fourcc, fps_src if fps_src > 0 else 30, (width, height))
        print(f"[INFO] Menyimpan video ke: {out_name}")

    # ---- Loop inferensi ----
    frame_count  = 0
    total_infer  = 0.0
    screenshot_n = 0

    print("\n[START] Inferensi dimulai...\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if not is_camera:
                    print("[INFO] Video selesai.")
                break

            # Inferensi
            t0      = time.time()
            results = model.predict(
                source=frame,
                conf=conf,
                iou=iou,
                imgsz=IMG_SIZE,
                device=device,
                verbose=False,
                stream=False,
            )
            infer_ms = (time.time() - t0) * 1000

            result       = results[0]
            total_infer += infer_ms
            frame_count += 1

            # Hitung FPS
            fps_infer = 1000.0 / infer_ms if infer_ms > 0 else 0

            # Gambar dengan anotasi
            annotated = result.plot()

            # Overlay info di frame
            n_det = len(result.boxes)
            cv2.putText(annotated, f"FPS: {fps_infer:.1f}",    (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(annotated, f"Obj: {n_det}",            (10, 65),  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
            cv2.putText(annotated, f"Conf: {conf:.2f}",        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

            # Tampilkan
            cv2.imshow("TensorRT Inference - Jetson Nano", annotated)

            # Log ke terminal setiap 30 frame
            if frame_count % 30 == 0:
                avg_infer = total_infer / frame_count
                print(f"  Frame {frame_count:5d} | Infer: {infer_ms:6.1f}ms | Avg: {avg_infer:6.1f}ms | FPS: {fps_infer:.1f} | Deteksi: {n_det}")

            # Simpan jika --save
            if writer:
                writer.write(annotated)

            # Keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[INFO] Dihentikan oleh pengguna.")
                break
            elif key == ord('s') and is_camera:
                screenshot_n += 1
                fname = f"screenshot_{screenshot_n:04d}.jpg"
                cv2.imwrite(fname, annotated)
                print(f"  [SCREENSHOT] Disimpan: {fname}")

    except KeyboardInterrupt:
        print("\n[INFO] Dihentikan (Ctrl+C).")

    finally:
        cap.release()
        if writer:
            writer.release()
            print(f"[SAVED] Video disimpan.")
        cv2.destroyAllWindows()

    # Ringkasan
    if frame_count > 0:
        avg_ms  = total_infer / frame_count
        avg_fps = 1000.0 / avg_ms if avg_ms > 0 else 0
        print(f"\n{'='*50}")
        print(f"  Total Frame : {frame_count}")
        print(f"  Avg Infer   : {avg_ms:.1f} ms/frame")
        print(f"  Avg FPS     : {avg_fps:.1f}")
        print(f"{'='*50}")


# ============================================================================
#  MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Inferensi dengan TensorRT Engine di Jetson Nano"
    )
    parser.add_argument(
        "--engine", "-e", type=str, default=DEFAULT_ENGINE,
        help=f"Path ke file .engine (default: {DEFAULT_ENGINE})"
    )
    parser.add_argument(
        "--source", "-s", type=str, default="0",
        help="Sumber input: 0=kamera, path gambar/video (default: 0)"
    )
    parser.add_argument(
        "--conf", "-c", type=float, default=DEFAULT_CONF,
        help=f"Confidence threshold (default: {DEFAULT_CONF})"
    )
    parser.add_argument(
        "--iou", type=float, default=DEFAULT_IOU,
        help=f"IoU threshold untuk NMS (default: {DEFAULT_IOU})"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Simpan hasil inferensi (gambar/video)"
    )
    parser.add_argument(
        "--device", "-d", type=str, default="auto",
        help="Device: 'auto' (default), '0' (GPU), 'cpu'"
    )

    args = parser.parse_args()

    # Pilih device
    device = get_device(args.device)

    print("=" * 60)
    print("  TensorRT Inference — NVIDIA Jetson Nano")
    print("=" * 60)
    print(f"  Engine  : {args.engine}")
    print(f"  Source  : {args.source}")
    print(f"  Conf    : {args.conf}")
    print(f"  IoU     : {args.iou}")
    print(f"  Save    : {args.save}")
    print(f"  Device  : {device}")
    print("=" * 60)

    # Load model
    model = load_model(args.engine, device)

    # Deteksi tipe sumber
    source = args.source
    is_image = (
        not str(source).isdigit()
        and os.path.isfile(source)
        and source.lower().split(".")[-1] in ["jpg", "jpeg", "png", "bmp", "tiff"]
    )

    if is_image:
        run_on_image(model, source, args.conf, args.iou, args.save, device)
    else:
        run_on_video_or_camera(model, source, args.conf, args.iou, args.save, device)


if __name__ == "__main__":
    main()
