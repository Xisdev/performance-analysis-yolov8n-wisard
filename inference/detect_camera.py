"""
==============================================================================
  Deteksi Objek Real-Time via Kamera (CPU/GPU)
==============================================================================
  Menjalankan YOLOv8 untuk deteksi manusia secara real-time
  menggunakan kamera (webcam/kamera eksternal).

  Cara pakai:
      python detect_camera.py
      python detect_camera.py --cam 1 --conf 0.35
      python detect_camera.py --model models/best_yolov8n.pt --gpu
==============================================================================
"""

import os
import time
import cv2
import torch
import argparse
from ultralytics import YOLO
from multiprocessing import freeze_support

# =====================================================================
#  KONFIGURASI DEFAULT
# =====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

DEFAULT_MODEL = os.path.join(ROOT_DIR, "models", "best_yolov8n.pt")

CAMERA_INDEX = 0
CAM_WIDTH = 1280
CAM_HEIGHT = 720
IMGSZ = 416
CONF = 0.35
IOU = 0.45
DEVICE = "cpu"

# --- Warna & font overlay ---
COLOR_BOX = (0, 255, 80)
COLOR_LABEL = (0, 0, 0)
COLOR_FPS = (0, 220, 255)
COLOR_STATUS = (80, 200, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# =====================================================================
#  FUNGSI BANTU
# =====================================================================
def draw_detections(frame, result, class_names):
    """Gambar bounding box + label di atas frame."""
    if result.boxes is None or len(result.boxes) == 0:
        return frame, 0

    count = len(result.boxes)
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = f"{class_names.get(cls, str(cls))} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOX, 2)

        (tw, th), baseline = cv2.getTextSize(label, FONT, 0.55, 1)
        cv2.rectangle(frame,
                      (x1, max(0, y1 - th - baseline - 4)),
                      (x1 + tw + 4, y1),
                      COLOR_BOX, -1)

        cv2.putText(frame, label,
                    (x1 + 2, max(th + 2, y1 - baseline - 2)),
                    FONT, 0.55, COLOR_LABEL, 1, cv2.LINE_AA)

    return frame, count


def draw_hud(frame, fps, obj_count, device_str, cam_idx):
    """Tampilkan HUD (FPS, jumlah objek, info device) di frame."""
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    fps_text = f"FPS: {fps:5.1f}"
    cv2.putText(frame, fps_text, (10, 33), FONT, 0.85, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, fps_text, (10, 33), FONT, 0.85, COLOR_FPS, 2, cv2.LINE_AA)

    obj_text = f"Deteksi: {obj_count} objek"
    cv2.putText(frame, obj_text, (w // 2 - 80, 33), FONT, 0.75, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, obj_text, (w // 2 - 80, 33), FONT, 0.75, COLOR_STATUS, 2, cv2.LINE_AA)

    info_text = f"CAM:{cam_idx} | {device_str.upper()}"
    (iw, _), _ = cv2.getTextSize(info_text, FONT, 0.60, 1)
    cv2.putText(frame, info_text, (w - iw - 10, 33), FONT, 0.60, (180, 180, 180), 1, cv2.LINE_AA)

    hints = ["[Q] Keluar", "[S] Screenshot", "[R] Reset FPS"]
    for i, hint in enumerate(hints):
        cv2.putText(frame, hint, (10, h - 15 - i * 22), FONT, 0.5, (120, 120, 120), 1, cv2.LINE_AA)

    return frame


def list_cameras(max_check=6):
    """Cari indeks kamera yang tersedia."""
    available = []
    print("[INFO] Mendeteksi kamera yang tersedia...")
    for i in range(max_check):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
                print(f"  ✓ Kamera indeks {i} tersedia")
        cap.release()
    if not available:
        print("  ✗ Tidak ada kamera yang terdeteksi!")
    return available


# =====================================================================
#  MAIN
# =====================================================================
def jalankan_deteksi_kamera(model_path, cam_idx, device, imgsz, conf, iou,
                             save_output="", show_window=True):

    torch.set_num_threads(os.cpu_count() or 4)
    print("=" * 65)
    print("  DETEKSI OBJEK REAL-TIME — KAMERA")
    print("=" * 65)
    print(f"  PyTorch     : {torch.__version__}")
    print(f"  CPU Threads : {torch.get_num_threads()}")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"[ERROR] Model tidak ditemukan:\n  {model_path}")

    print(f"\n[INFO] Memuat model : {os.path.basename(model_path)}")
    model = YOLO(model_path)
    class_names = model.names
    print(f"[INFO] Kelas model  : {list(class_names.values())}")

    available_cams = list_cameras()
    if cam_idx not in available_cams:
        if available_cams:
            print(f"[WARN] Kamera indeks {cam_idx} tidak tersedia. Menggunakan indeks {available_cams[0]}.")
            cam_idx = available_cams[0]
        else:
            raise RuntimeError("[ERROR] Tidak ada kamera yang dapat dibuka.")

    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        raise RuntimeError(f"[ERROR] Gagal membuka kamera indeks {cam_idx}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n  Model  : {os.path.basename(model_path)}")
    print(f"  Kamera : indeks {cam_idx}  |  {actual_w}x{actual_h}")
    print(f"  imgsz  : {imgsz}  |  conf: {conf}  |  iou: {iou}")
    print(f"  Device : {device}")
    print("\n  Tekan Q atau ESC untuk keluar\n")

    win_name = "YOLOv8 | Deteksi Kamera"
    if show_window:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, min(actual_w, 1280), min(actual_h, 720))

    frame_idx = 0
    total_detect = 0
    fps_display = 0.0
    t_prev = time.perf_counter()
    t_start = t_prev

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.03)
                continue

            frame_idx += 1

            results = model.predict(
                source=frame, imgsz=imgsz, conf=conf, iou=iou,
                device=device, verbose=False,
            )

            t_now = time.perf_counter()
            instant_fps = 1.0 / max(t_now - t_prev, 1e-6)
            fps_display = 0.15 * instant_fps + 0.85 * fps_display
            t_prev = t_now

            annotated, det_count = draw_detections(frame, results[0], class_names)
            annotated = draw_hud(annotated, fps_display, det_count, device, cam_idx)
            total_detect += det_count

            if show_window:
                cv2.imshow(win_name, annotated)

            if frame_idx % 60 == 0:
                elapsed = t_now - t_start
                print(f"  Frame {frame_idx:>6} | FPS: {fps_display:5.1f} | "
                      f"Deteksi: {total_detect} | Waktu: {elapsed:.1f}s")

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                print("\n[INFO] Dihentikan oleh pengguna.")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Dihentikan dengan Ctrl+C.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    total_time = time.perf_counter() - t_start
    avg_fps = frame_idx / total_time if total_time > 0 else 0

    print(f"\n{'=' * 65}")
    print(f"  Total frame  : {frame_idx}")
    print(f"  Total deteksi: {total_detect}")
    print(f"  Rata-rata FPS: {avg_fps:.2f}")
    print(f"  Total waktu  : {total_time:.1f} detik")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deteksi objek real-time via kamera")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Path ke model .pt")
    parser.add_argument("--cam", type=int, default=CAMERA_INDEX,
                        help=f"Indeks kamera (default: {CAMERA_INDEX})")
    parser.add_argument("--gpu", action="store_true",
                        help="Gunakan GPU (default: CPU)")
    parser.add_argument("--conf", type=float, default=CONF,
                        help=f"Confidence threshold (default: {CONF})")
    parser.add_argument("--imgsz", type=int, default=IMGSZ,
                        help=f"Ukuran inferensi (default: {IMGSZ})")
    parser.add_argument("--iou", type=float, default=IOU,
                        help=f"NMS IoU threshold (default: {IOU})")
    parser.add_argument("--no-window", action="store_true",
                        help="Jalankan tanpa window preview")
    args = parser.parse_args()

    device = "0" if args.gpu else "cpu"

    freeze_support()
    jalankan_deteksi_kamera(
        model_path=args.model,
        cam_idx=args.cam,
        device=device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        show_window=not args.no_window,
    )
