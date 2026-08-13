import os
import time
import cv2
import torch
from ultralytics import YOLO
from multiprocessing import freeze_support

# =====================================================================
#  KONFIGURASI
# =====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH  = r"D:\RISET\drone-wisard\runs_yolov8n_wisard\runs_yolov8n_wisard\yolov8n_optimized\weights\best.pt"
VIDEO_INPUT = r"D:\RISET\drone-wisard\vidtest.mp4"
VIDEO_OUTPUT= r"D:\RISET\drone-wisard\result.mp4"

# --- Inference settings (ringan untuk CPU) ---
IMGSZ   = 320       # resolusi kecil → lebih cepat
CONF    = 0.25      # confidence threshold
IOU     = 0.5       # NMS IoU threshold
DEVICE  = "cpu"

# --- warna & font overlay ---
COLOR_BOX   = (0, 255, 80)      # hijau neon untuk bounding box
COLOR_LABEL = (0, 0, 0)         # teks label (hitam)
COLOR_FPS   = (0, 220, 255)     # kuning untuk FPS overlay
FONT        = cv2.FONT_HERSHEY_SIMPLEX


# =====================================================================
#  FUNGSI BANTU
# =====================================================================
def draw_detections(frame, result, class_names):
    """Gambar bounding box + label di atas frame."""
    if result.boxes is None or len(result.boxes) == 0:
        return frame

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf  = float(box.conf[0])
        cls   = int(box.cls[0])
        label = f"{class_names[cls]} {conf:.2f}"

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOX, 2)

        # Label background
        (tw, th), baseline = cv2.getTextSize(label, FONT, 0.55, 1)
        cv2.rectangle(frame,
                      (x1, y1 - th - baseline - 4),
                      (x1 + tw + 4, y1),
                      COLOR_BOX, -1)

        # Label text
        cv2.putText(frame, label,
                    (x1 + 2, y1 - baseline - 2),
                    FONT, 0.55, COLOR_LABEL, 1, cv2.LINE_AA)

    return frame


def draw_fps_overlay(frame, fps, frame_idx, total_frames):
    """Tampilkan FPS + progress di pojok kiri atas."""
    h, w = frame.shape[:2]

    # FPS
    fps_text = f"FPS: {fps:5.1f}"
    cv2.putText(frame, fps_text,
                (12, 32), FONT, 0.9, (0, 0, 0), 3, cv2.LINE_AA)          # shadow
    cv2.putText(frame, fps_text,
                (12, 32), FONT, 0.9, COLOR_FPS, 2, cv2.LINE_AA)

    # Progress frame
    prog_text = f"Frame: {frame_idx}/{total_frames}"
    cv2.putText(frame, prog_text,
                (12, 60), FONT, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, prog_text,
                (12, 60), FONT, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    # Progress bar di bawah frame
    if total_frames > 0:
        bar_w = int(w * frame_idx / total_frames)
        cv2.rectangle(frame, (0, h - 6), (w, h), (50, 50, 50), -1)
        cv2.rectangle(frame, (0, h - 6), (bar_w, h), COLOR_FPS, -1)

    return frame


# =====================================================================
#  MAIN
# =====================================================================
def jalankan_deteksi_video():
    # --- Info environment ---
    torch.set_num_threads(os.cpu_count() or 4)
    print(f"PyTorch   : {torch.__version__}")
    print(f"CPU Threads: {torch.get_num_threads()}")
    import ultralytics
    print(f"Ultralytics: {ultralytics.__version__}")
    print()

    # --- Cek file ---
    for path, label in [(MODEL_PATH, "Model"), (VIDEO_INPUT, "Video input")]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"[ERROR] {label} tidak ditemukan:\n  {path}")

    # --- Load model ---
    print(f"Memuat model: {os.path.basename(MODEL_PATH)}")
    model = YOLO(MODEL_PATH)
    class_names = model.names
    print(f"Kelas       : {list(class_names.values())}")

    # --- Buka video input ---
    cap = cv2.VideoCapture(VIDEO_INPUT)
    if not cap.isOpened():
        raise RuntimeError(f"[ERROR] Tidak bisa membuka video:\n  {VIDEO_INPUT}")

    src_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_fr  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print()
    print("=" * 60)
    print("  DETEKSI VIDEO")
    print("=" * 60)
    print(f"  Input  : {VIDEO_INPUT}")
    print(f"  Output : {VIDEO_OUTPUT}")
    print(f"  Resolusi: {src_w}x{src_h}  @{src_fps:.1f} FPS  ({total_fr} frames)")
    print(f"  imgsz  : {IMGSZ}  |  conf: {CONF}  |  iou: {IOU}")
    print(f"  Device : {DEVICE}")
    print("=" * 60)
    print()

    # --- Siapkan VideoWriter ---
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, src_fps, (src_w, src_h))

    # --- Loop deteksi ---
    frame_idx   = 0
    fps_display = 0.0
    fps_alpha   = 0.2          # smoothing EWMA
    t_prev      = time.perf_counter()
    t_start     = t_prev

    print("Memulai inferensi... (tekan Ctrl+C untuk berhenti)")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # --- Inferensi ---
            results = model.predict(
                source=frame,
                imgsz=IMGSZ,
                conf=CONF,
                iou=IOU,
                device=DEVICE,
                verbose=False,
            )

            # --- Hitung FPS (EWMA) ---
            t_now = time.perf_counter()
            instant_fps = 1.0 / max(t_now - t_prev, 1e-6)
            fps_display = fps_alpha * instant_fps + (1 - fps_alpha) * fps_display
            t_prev = t_now

            # --- Gambar deteksi + overlay ---
            annotated = draw_detections(frame, results[0], class_names)
            annotated = draw_fps_overlay(annotated, fps_display, frame_idx, total_fr)

            # --- Tulis frame ---
            out.write(annotated)

            # --- Progress ke terminal setiap 30 frame ---
            if frame_idx % 30 == 0 or frame_idx == 1:
                pct = frame_idx / total_fr * 100 if total_fr > 0 else 0
                elapsed = t_now - t_start
                eta = (elapsed / frame_idx) * (total_fr - frame_idx) if frame_idx > 0 else 0
                print(f"  Frame {frame_idx:>5}/{total_fr}  ({pct:5.1f}%)  "
                      f"FPS: {fps_display:5.1f}  ETA: {eta:6.1f}s")

    except KeyboardInterrupt:
        print("\n[INFO] Dihentikan oleh pengguna.")

    finally:
        cap.release()
        out.release()

    # --- Ringkasan akhir ---
    total_time = time.perf_counter() - t_start
    avg_fps = frame_idx / total_time if total_time > 0 else 0

    print()
    print("=" * 60)
    print("  SELESAI")
    print("=" * 60)
    print(f"  Total frame diproses : {frame_idx}")
    print(f"  Total waktu          : {total_time:.1f} detik")
    print(f"  Rata-rata FPS        : {avg_fps:.2f}")
    print(f"  Hasil disimpan di    : {VIDEO_OUTPUT}")
    print("=" * 60)


# =====================================================================
#  ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    freeze_support()
    jalankan_deteksi_video()
