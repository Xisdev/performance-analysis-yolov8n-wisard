"""
detect_camera_gpu.py
====================
Deteksi objek real-time YOLOv8n + kamera eksternal
menggunakan ONNX Runtime + DirectML (AMD GPU di Windows).

Pipeline:
  1. Export model .pt → .onnx (sekali, otomatis)
  2. Load sesi ONNX Runtime dengan DmlExecutionProvider (AMD GPU)
  3. Loop kamera → preprocess → inferensi ONNX → postprocess (NMS) → tampil

Instalasi (sekali):
  pip install onnxruntime-directml
  pip install ultralytics opencv-python

Menjalankan:
  python detect_camera_gpu.py              # kamera 1, GPU AMD
  python detect_camera_gpu.py --cam 0      # kamera 0
  python detect_camera_gpu.py --cpu        # paksa CPU
  python detect_camera_gpu.py --imgsz 320  # inferensi lebih cepat
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
from multiprocessing import freeze_support

# =====================================================================
#  KONFIGURASI DEFAULT
# =====================================================================
MODEL_PT   = r"D:\RISET\drone-wisard\runs_yolov8n_wisard\runs_yolov8n_wisard\yolov8n_optimized\weights\best.pt"
# Path ONNX diturunkan otomatis dari MODEL_PT
MODEL_ONNX = os.path.splitext(MODEL_PT)[0] + "_camera.onnx"

CAMERA_INDEX = 1
CAM_WIDTH    = 1280
CAM_HEIGHT   = 720

IMGSZ        = 416       # harus kelipatan 32; coba 320 jika lambat
CONF         = 0.35
IOU          = 0.45

SAVE_OUTPUT  = ""        # contoh: r"D:\RISET\drone-wisard\output_gpu.mp4"

# Warna HUD
COLOR_BOX    = (0, 255, 80)
COLOR_LABEL  = (0, 0, 0)
COLOR_FPS    = (0, 220, 255)
COLOR_STATUS = (80, 200, 255)
COLOR_GPU    = (200, 100, 255)
FONT         = cv2.FONT_HERSHEY_SIMPLEX


# =====================================================================
#  EXPORT KE ONNX (jika belum ada)
# =====================================================================
def export_onnx_if_needed(pt_path: str, onnx_path: str, imgsz: int) -> None:
    if os.path.isfile(onnx_path):
        print(f"[INFO] ONNX model sudah ada: {os.path.basename(onnx_path)}")
        return
    print(f"[INFO] Mengekspor model ke ONNX (imgsz={imgsz}) ...")
    print(f"       Ini hanya dilakukan sekali. Mohon tunggu...")
    from ultralytics import YOLO
    model = YOLO(pt_path)
    model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=False,
        simplify=True,
        opset=17,
    )
    # Ultralytics menyimpan di folder yang sama dengan nama model
    default_onnx = os.path.splitext(pt_path)[0] + ".onnx"
    if os.path.isfile(default_onnx) and default_onnx != onnx_path:
        import shutil
        shutil.move(default_onnx, onnx_path)
    if not os.path.isfile(onnx_path):
        # fallback: coba path standar
        if os.path.isfile(default_onnx):
            MODEL_ONNX = default_onnx  # noqa: F841
    print(f"[INFO] ONNX tersimpan: {onnx_path}")


# =====================================================================
#  ONNX RUNTIME SESSION
# =====================================================================
def create_ort_session(onnx_path: str, use_gpu: bool):
    """
    Buat sesi ONNX Runtime.
    Urutan provider: DmlExecutionProvider (AMD GPU) → CPUExecutionProvider.
    """
    import onnxruntime as ort

    available = [p.upper() for p in ort.get_available_providers()]
    print(f"[INFO] ONNX Runtime providers tersedia: {ort.get_available_providers()}")

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads      = os.cpu_count() or 4

    if use_gpu and "DMLEXECUTIONPROVIDER" in available:
        providers = [
            ("DmlExecutionProvider", {"device_id": 0}),
            "CPUExecutionProvider",
        ]
        device_label = "AMD GPU (DirectML)"
    else:
        providers = ["CPUExecutionProvider"]
        device_label = "CPU"
        if use_gpu:
            print("[WARN] DmlExecutionProvider tidak tersedia, fallback CPU.")

    session = ort.InferenceSession(onnx_path, sess_options=opts, providers=providers)
    actual_provider = session.get_providers()[0]
    print(f"[INFO] Provider aktif : {actual_provider}")
    print(f"[INFO] Device digunakan: {device_label}")
    return session, device_label


# =====================================================================
#  PRE & POST PROCESSING
# =====================================================================
def preprocess(frame: np.ndarray, imgsz: int):
    """
    Resize + normalize frame untuk input ONNX YOLOv8.
    Returns:
        blob      – float32 array [1, 3, imgsz, imgsz]
        ratio     – (rx, ry) skala dari imgsz ke frame asli
        pad       – (pad_x, pad_y) padding letterbox
    """
    h, w = frame.shape[:2]

    # ── Letterbox resize ──
    scale = min(imgsz / w, imgsz / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # ── Padding ke imgsz×imgsz ──
    pad_x = (imgsz - new_w) // 2
    pad_y = (imgsz - new_h) // 2
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    # ── Normalize + CHW + batch ──
    blob = canvas.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis, :]   # [1, 3, H, W]

    return blob, scale, (pad_x, pad_y)


def nms_boxes(boxes_xyxy, scores, iou_thresh):
    """NMS sederhana menggunakan cv2.dnn.NMSBoxes."""
    if len(boxes_xyxy) == 0:
        return []
    # cv2 memerlukan x,y,w,h
    boxes_xywh = []
    for x1, y1, x2, y2 in boxes_xyxy:
        boxes_xywh.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
    idxs = cv2.dnn.NMSBoxes(boxes_xywh, [float(s) for s in scores],
                              score_threshold=0.0, nms_threshold=iou_thresh)
    if len(idxs) == 0:
        return []
    return [int(i) for i in idxs.flatten()]


def postprocess(output: np.ndarray, conf_thresh: float, iou_thresh: float,
                orig_shape: tuple, imgsz: int, scale: float, pad: tuple):
    """
    Parse output YOLOv8 ONNX.
    Output shape: [1, num_classes+4, 8400]  →  [8400, 4+num_classes]
    """
    pred = output[0]                    # [4+nc, 8400]
    pred = pred.T                       # [8400, 4+nc]

    boxes_cxywh = pred[:, :4]          # cx, cy, w, h (dalam koordinat imgsz)
    class_scores = pred[:, 4:]         # [8400, nc]

    max_scores = class_scores.max(axis=1)
    class_ids  = class_scores.argmax(axis=1)

    # Filter confidence
    mask = max_scores >= conf_thresh
    if not mask.any():
        return [], [], []

    boxes_f  = boxes_cxywh[mask]
    scores_f = max_scores[mask]
    cls_f    = class_ids[mask]

    # Konversi cx,cy,w,h → x1,y1,x2,y2 (koordinat dalam ruang imgsz)
    pad_x, pad_y = pad
    oh, ow = orig_shape[:2]

    x1 = (boxes_f[:, 0] - boxes_f[:, 2] / 2 - pad_x) / scale
    y1 = (boxes_f[:, 1] - boxes_f[:, 3] / 2 - pad_y) / scale
    x2 = (boxes_f[:, 0] + boxes_f[:, 2] / 2 - pad_x) / scale
    y2 = (boxes_f[:, 1] + boxes_f[:, 3] / 2 - pad_y) / scale

    # Clip ke batas frame
    x1 = np.clip(x1, 0, ow).astype(int)
    y1 = np.clip(y1, 0, oh).astype(int)
    x2 = np.clip(x2, 0, ow).astype(int)
    y2 = np.clip(y2, 0, oh).astype(int)

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # NMS per kelas
    keep_all = []
    for cid in np.unique(cls_f):
        mask_c = cls_f == cid
        idx_c  = np.where(mask_c)[0]
        keep   = nms_boxes(boxes_xyxy[idx_c], scores_f[idx_c], iou_thresh)
        keep_all.extend(idx_c[keep])

    if not keep_all:
        return [], [], []

    keep_all = sorted(keep_all)
    return (boxes_xyxy[keep_all].tolist(),
            scores_f[keep_all].tolist(),
            cls_f[keep_all].tolist())


# =====================================================================
#  DRAW
# =====================================================================
def draw_detections(frame, boxes, scores, class_ids, class_names):
    count = len(boxes)
    for (x1, y1, x2, y2), conf, cls in zip(boxes, scores, class_ids):
        label = f"{class_names.get(int(cls), str(cls))} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOX, 2)
        (tw, th), bl = cv2.getTextSize(label, FONT, 0.55, 1)
        cv2.rectangle(frame,
                      (x1, max(0, y1 - th - bl - 4)),
                      (x1 + tw + 4, y1), COLOR_BOX, -1)
        cv2.putText(frame, label,
                    (x1 + 2, max(th + 2, y1 - bl - 2)),
                    FONT, 0.55, COLOR_LABEL, 1, cv2.LINE_AA)
    return frame, count


def draw_hud(frame, fps, obj_count, device_label, cam_idx, conf_thresh):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 58), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

    # FPS
    fps_text = f"FPS: {fps:5.1f}"
    cv2.putText(frame, fps_text, (10, 38), FONT, 0.85, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, fps_text, (10, 38), FONT, 0.85, COLOR_FPS, 2, cv2.LINE_AA)

    # Jumlah objek (tengah)
    obj_text = f"Deteksi: {obj_count}"
    (ow, _), _ = cv2.getTextSize(obj_text, FONT, 0.75, 2)
    cv2.putText(frame, obj_text, (w // 2 - ow // 2, 38),
                FONT, 0.75, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, obj_text, (w // 2 - ow // 2, 38),
                FONT, 0.75, COLOR_STATUS, 2, cv2.LINE_AA)

    # Device info (kanan)
    dev_text = f"{device_label} | CAM:{cam_idx}"
    (dw, _), _ = cv2.getTextSize(dev_text, FONT, 0.55, 1)
    cv2.putText(frame, dev_text, (w - dw - 10, 28),
                FONT, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, dev_text, (w - dw - 10, 28),
                FONT, 0.55, COLOR_GPU, 1, cv2.LINE_AA)

    # Conf (kanan bawah header)
    conf_text = f"conf: {conf_thresh:.2f}"
    (cw, _), _ = cv2.getTextSize(conf_text, FONT, 0.48, 1)
    cv2.putText(frame, conf_text, (w - cw - 10, 50),
                FONT, 0.48, (160, 160, 160), 1, cv2.LINE_AA)

    # Kontrol hint
    hints = ["[Q/ESC] Keluar", "[S] Screenshot", "[R] Reset FPS", "[+/-] Conf"]
    for i, hint in enumerate(hints):
        cv2.putText(frame, hint, (10, h - 15 - i * 22),
                    FONT, 0.46, (100, 100, 100), 1, cv2.LINE_AA)
    return frame


# =====================================================================
#  KAMERA
# =====================================================================
def list_cameras(max_check=6):
    available = []
    print("[INFO] Mendeteksi kamera yang tersedia...")
    for i in range(max_check):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
                print(f"  ✓ Kamera indeks {i} tersedia")
        cap.release()
    if not available:
        print("  ✗ Tidak ada kamera terdeteksi!")
    return available


def open_camera(cam_idx, width, height):
    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS,          30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    return cap


# =====================================================================
#  MAIN
# =====================================================================
def jalankan_deteksi_gpu(cam_idx=CAMERA_INDEX,
                         save_output=SAVE_OUTPUT,
                         show_window=True,
                         use_gpu=True,
                         imgsz=IMGSZ,
                         conf=CONF):

    print("=" * 65)
    print("  DETEKSI REAL-TIME – AMD GPU (ONNX Runtime + DirectML)")
    print("=" * 65)

    # ── Cek model .pt ──
    if not os.path.isfile(MODEL_PT):
        raise FileNotFoundError(f"[ERROR] Model tidak ditemukan:\n  {MODEL_PT}")

    # ── Export ke ONNX jika belum ada ──
    onnx_path = os.path.splitext(MODEL_PT)[0] + f"_imgsz{imgsz}.onnx"
    export_onnx_if_needed(MODEL_PT, onnx_path, imgsz)

    # ── Ambil nama kelas dari model .pt ──
    from ultralytics import YOLO as _YOLO
    _tmp_model  = _YOLO(MODEL_PT)
    class_names = _tmp_model.names
    del _tmp_model
    print(f"[INFO] Kelas: {list(class_names.values())}")
    print()

    # ── Buat sesi ONNX Runtime ──
    session, device_label = create_ort_session(onnx_path, use_gpu)
    input_name  = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"[INFO] Input  : {input_name}  shape: {session.get_inputs()[0].shape}")
    print(f"[INFO] Output : {output_name} shape: {session.get_outputs()[0].shape}")
    print()

    # ── Kamera ──
    available_cams = list_cameras()
    if cam_idx not in available_cams:
        if available_cams:
            print(f"[WARN] Kamera {cam_idx} tidak tersedia, menggunakan {available_cams[0]}.")
            cam_idx = available_cams[0]
        else:
            raise RuntimeError("[ERROR] Tidak ada kamera yang dapat dibuka.")

    print(f"\n[INFO] Membuka kamera indeks {cam_idx} ({CAM_WIDTH}x{CAM_HEIGHT}) ...")
    cap = open_camera(cam_idx, CAM_WIDTH, CAM_HEIGHT)
    if cap is None or not cap.isOpened():
        raise RuntimeError(f"[ERROR] Gagal membuka kamera {cam_idx}.")

    actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[INFO] Resolusi aktual : {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")

    # ── VideoWriter ──
    out_writer = None
    if save_output:
        fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(save_output, fourcc, actual_fps,
                                     (actual_w, actual_h))
        print(f"[INFO] Menyimpan output ke : {save_output}")

    # ── Window ──
    win_name = f"YOLOv8n | {device_label} | Real-time Deteksi"
    if show_window:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, min(actual_w, 1280), min(actual_h, 720))

    print()
    print("=" * 65)
    print(f"  Model   : {os.path.basename(MODEL_PT)}")
    print(f"  Device  : {device_label}")
    print(f"  Kamera  : indeks {cam_idx}  |  {actual_w}x{actual_h}")
    print(f"  imgsz   : {imgsz}  |  conf: {conf}  |  iou: {IOU}")
    print("=" * 65)
    print()
    print("  Kontrol:")
    print("    Q / ESC  → Keluar")
    print("    S        → Simpan screenshot")
    print("    R        → Reset counter FPS")
    print("    +/-      → Naikkan/Turunkan confidence threshold")
    print()
    print("  Tekan Enter untuk mulai inferensi...")
    input()

    # ── Warm-up ONNX session (1 frame dummy) ──
    print("[INFO] Warm-up ONNX session ...")
    dummy = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
    session.run([output_name], {input_name: dummy})
    print("[INFO] Siap!")
    print()

    # ── Loop utama ──
    frame_idx    = 0
    total_detect = 0
    fps_display  = 0.0
    fps_alpha    = 0.15
    conf_thresh  = conf
    t_prev       = time.perf_counter()
    t_start      = t_prev
    screenshot_n = 0
    skip_count   = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                skip_count += 1
                if skip_count > 30:
                    print("[ERROR] Terlalu banyak frame gagal. Berhenti.")
                    break
                time.sleep(0.02)
                continue
            skip_count = 0
            frame_idx += 1

            # ── Preprocess ──
            blob, scale, pad = preprocess(frame, imgsz)

            # ── Inferensi ONNX ──
            output = session.run([output_name], {input_name: blob})[0]

            # ── Postprocess ──
            boxes, scores, cls_ids = postprocess(
                output, conf_thresh, IOU,
                frame.shape, imgsz, scale, pad
            )

            # ── Hitung FPS ──
            t_now       = time.perf_counter()
            instant_fps = 1.0 / max(t_now - t_prev, 1e-6)
            fps_display = fps_alpha * instant_fps + (1 - fps_alpha) * fps_display
            t_prev      = t_now

            # ── Gambar ──
            annotated, det_count = draw_detections(
                frame.copy(), boxes, scores, cls_ids, class_names
            )
            annotated = draw_hud(annotated, fps_display, det_count,
                                  device_label, cam_idx, conf_thresh)
            total_detect += det_count

            # ── Tulis output ──
            if out_writer:
                out_writer.write(annotated)

            if show_window:
                cv2.imshow(win_name, annotated)

            # ── Log terminal ──
            if frame_idx % 60 == 0:
                elapsed = t_now - t_start
                print(f"  Frame {frame_idx:>6} | FPS: {fps_display:5.1f} | "
                      f"Deteksi: {total_detect} | conf: {conf_thresh:.2f} | "
                      f"Waktu: {elapsed:.1f}s")

            # ── Keyboard handler ──
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                print("\n[INFO] Dihentikan oleh pengguna.")
                break
            elif key in (ord('s'), ord('S')):
                screenshot_n += 1
                fname = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    f"screenshot_gpu_{screenshot_n:03d}.jpg"
                )
                cv2.imwrite(fname, annotated)
                print(f"[INFO] Screenshot: {fname}")
            elif key in (ord('r'), ord('R')):
                fps_display = 0.0
                t_prev      = time.perf_counter()
                print("[INFO] FPS counter direset.")
            elif key in (ord('+'), ord('=')):
                conf_thresh = min(0.95, round(conf_thresh + 0.05, 2))
                print(f"[INFO] Confidence → {conf_thresh:.2f}")
            elif key == ord('-'):
                conf_thresh = max(0.05, round(conf_thresh - 0.05, 2))
                print(f"[INFO] Confidence → {conf_thresh:.2f}")

    except KeyboardInterrupt:
        print("\n[INFO] Dihentikan dengan Ctrl+C.")

    finally:
        cap.release()
        if out_writer:
            out_writer.release()
        cv2.destroyAllWindows()

    # ── Ringkasan ──
    total_time = time.perf_counter() - t_start
    avg_fps    = frame_idx / total_time if total_time > 0 else 0
    print()
    print("=" * 65)
    print("  SELESAI")
    print("=" * 65)
    print(f"  Device           : {device_label}")
    print(f"  Total frame      : {frame_idx}")
    print(f"  Total deteksi    : {total_detect}")
    print(f"  Rata-rata FPS    : {avg_fps:.2f}")
    print(f"  Total waktu      : {total_time:.1f} detik")
    if save_output:
        print(f"  Video tersimpan  : {save_output}")
    print("=" * 65)


# =====================================================================
#  ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deteksi real-time YOLOv8n – AMD GPU via ONNX Runtime + DirectML"
    )
    parser.add_argument("--cam",       type=int,   default=CAMERA_INDEX,
                        help=f"Indeks kamera (default: {CAMERA_INDEX})")
    parser.add_argument("--save",      type=str,   default=SAVE_OUTPUT,
                        help="Path file output video (kosong = tidak disimpan)")
    parser.add_argument("--no-window", action="store_true",
                        help="Jalankan tanpa window preview")
    parser.add_argument("--conf",      type=float, default=CONF,
                        help=f"Confidence threshold (default: {CONF})")
    parser.add_argument("--imgsz",     type=int,   default=IMGSZ,
                        help=f"Ukuran inferensi ONNX (default: {IMGSZ}). Coba 320 jika lambat.")
    parser.add_argument("--cpu",       action="store_true",
                        help="Paksa CPU (nonaktifkan DirectML)")
    args = parser.parse_args()

    freeze_support()
    jalankan_deteksi_gpu(
        cam_idx     = args.cam,
        save_output = args.save,
        show_window = not args.no_window,
        use_gpu     = not args.cpu,
        imgsz       = args.imgsz,
        conf        = args.conf,
    )
