"""
==============================================================================
  Evaluasi Akurasi Model YOLOv8 pada Edge Device
==============================================================================
  Script ini mengevaluasi akurasi model (Precision, Recall, F1, mAP50)
  setelah dikonversi ke format deployment (.engine / .tflite).

  Dapat dijalankan di:
    - NVIDIA Jetson Nano (model .engine)
    - Raspberry Pi 4 (model .tflite)
    - PC (model .pt / .onnx)

  Cara pakai:
      python3 evaluate_accuracy.py --model best_fp16.engine \
          --images ./test_images/images/ --labels ./test_images/labels/

      # Batch semua model di folder
      python3 evaluate_accuracy.py --model-dir ./models/ \
          --images ./test_images/images/ --labels ./test_images/labels/

  Output:
      eval_accuracy_{model}_{device}.json
==============================================================================
"""

import os
import sys
import json
import time
import glob
import platform
import argparse
from pathlib import Path

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

DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.5    # IoU threshold untuk matching TP/FP
DEFAULT_NMS_IOU = 0.45
DEFAULT_IMGSZ = 640

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ============================================================================
#  FUNGSI EVALUASI
# ============================================================================

def xywh_to_xyxy(box):
    """Konversi YOLO xywh (normalized) ke xyxy (normalized)."""
    cx, cy, w, h = box
    return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]


def compute_iou(box1, box2):
    """Hitung IoU antara dua box dalam format xyxy."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def load_gt_labels(label_path):
    """Load label ground truth dari file YOLO txt."""
    labels = []
    if not os.path.exists(label_path):
        return labels
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                box = [float(x) for x in parts[1:5]]
                labels.append({'class': cls, 'bbox': box})
    return labels


def get_image_files(image_dir):
    """Ambil semua file gambar dari direktori."""
    files = []
    for f in sorted(os.listdir(image_dir)):
        if Path(f).suffix.lower() in IMG_EXTENSIONS:
            files.append(f)
    return files


def compute_ap(recalls, precisions):
    """
    Hitung Average Precision (AP) menggunakan interpolasi 11 titik.
    """
    recalls = np.array(recalls)
    precisions = np.array(precisions)

    # Tambahkan titik awal dan akhir
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([1.0], precisions, [0.0]))

    # Monotonically decreasing precision
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Interpolasi 11 titik
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        mask = recalls >= t
        if mask.any():
            ap += np.max(precisions[mask]) / 11.0

    return ap


# ============================================================================
#  EVALUASI UTAMA
# ============================================================================

def evaluate_model(model_path, image_dir, label_dir, output_dir,
                   conf=DEFAULT_CONF, iou_thresh=DEFAULT_IOU,
                   nms_iou=DEFAULT_NMS_IOU, imgsz=DEFAULT_IMGSZ,
                   device="cpu", device_name="unknown"):
    """
    Evaluasi akurasi satu model pada dataset test.

    Returns:
        dict: Hasil evaluasi (precision, recall, f1, mAP50, dll)
    """
    model_name = Path(model_path).stem

    print(f"\n{'=' * 70}")
    print(f"  EVALUASI AKURASI: {model_name}")
    print(f"  Device: {device_name}")
    print(f"{'=' * 70}")

    # Validasi
    if not os.path.exists(model_path):
        print(f"[ERROR] Model tidak ditemukan: {model_path}")
        return None

    image_files = get_image_files(image_dir)
    if not image_files:
        print(f"[ERROR] Tidak ada gambar di: {image_dir}")
        return None

    model_size = os.path.getsize(model_path) / (1024 * 1024)

    print(f"  Model    : {model_path}")
    print(f"  Size     : {model_size:.2f} MB")
    print(f"  Images   : {len(image_files)} gambar")
    print(f"  Labels   : {label_dir}")
    print(f"  Config   : imgsz={imgsz}, conf={conf}, iou_thresh={iou_thresh}")

    # Load model
    print(f"\n[1/3] Loading model...")
    t_load = time.time()

    try:
        model = YOLO(model_path, task="detect")
    except MemoryError:
        print(f"[ERROR] OUT OF MEMORY saat loading model!")
        return {
            'model_name': model_name,
            'model_size_mb': round(model_size, 2),
            'device': device_name,
            'status': 'OOM',
        }
    except Exception as e:
        print(f"[ERROR] Gagal load model: {e}")
        return None

    load_time = time.time() - t_load
    print(f"       Loaded dalam {load_time:.1f}s")

    # Evaluasi per gambar
    print(f"\n[2/3] Evaluasi per gambar...")

    total_tp, total_fp, total_fn = 0, 0, 0
    total_gt = 0
    all_pred_confs = []
    all_pred_is_tp = []
    all_ious = []

    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(image_dir, img_file)
        lbl_file = Path(img_file).stem + ".txt"
        lbl_path = os.path.join(label_dir, lbl_file)

        # Ground truth
        gt_labels = load_gt_labels(lbl_path)
        total_gt += len(gt_labels)

        # Prediksi
        try:
            results = model.predict(
                source=img_path,
                conf=conf, iou=nms_iou, imgsz=imgsz,
                device=device, verbose=False,
            )
        except MemoryError:
            print(f"  [ERROR] OOM pada gambar {idx + 1}")
            return {
                'model_name': model_name,
                'model_size_mb': round(model_size, 2),
                'device': device_name,
                'status': 'OOM',
                'oom_at_image': idx + 1,
            }
        except Exception as e:
            print(f"  [WARN] Error pada gambar {idx + 1}: {e}")
            total_fn += len(gt_labels)
            continue

        result = results[0]

        # Parse prediksi
        pred_boxes = []
        if result.boxes is not None and len(result.boxes) > 0:
            for i in range(len(result.boxes)):
                xyxy = result.boxes.xyxy[i].cpu().numpy()
                conf_val = float(result.boxes.conf[i].cpu().numpy())
                cls = int(result.boxes.cls[i].cpu().numpy())

                # Konversi xyxy ke xywh normalized
                img_h, img_w = result.orig_shape
                cx = ((xyxy[0] + xyxy[2]) / 2) / img_w
                cy = ((xyxy[1] + xyxy[3]) / 2) / img_h
                bw = (xyxy[2] - xyxy[0]) / img_w
                bh = (xyxy[3] - xyxy[1]) / img_h

                pred_boxes.append({
                    'class': cls,
                    'bbox': [float(cx), float(cy), float(bw), float(bh)],
                    'conf': conf_val,
                })

        # Matching TP/FP/FN
        gt_matched = [False] * len(gt_labels)
        sorted_preds = sorted(pred_boxes, key=lambda x: x['conf'], reverse=True)

        for pred in sorted_preds:
            pred_xyxy = xywh_to_xyxy(pred['bbox'])
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(gt_labels):
                if gt_matched[gt_idx]:
                    continue
                gt_xyxy = xywh_to_xyxy(gt['bbox'])
                iou = compute_iou(pred_xyxy, gt_xyxy)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_thresh and best_gt_idx >= 0:
                gt_matched[best_gt_idx] = True
                all_pred_is_tp.append(1)
                all_ious.append(best_iou)
                total_tp += 1
            else:
                all_pred_is_tp.append(0)
                total_fp += 1

            all_pred_confs.append(pred['conf'])

        # FN = GT yang tidak terdeteksi
        fn = sum(1 for m in gt_matched if not m)
        total_fn += fn

        # Progress
        if (idx + 1) % 50 == 0 or (idx + 1) == len(image_files):
            print(f"       {idx + 1}/{len(image_files)} gambar dievaluasi...")

    # Hitung metrik
    print(f"\n[3/3] Menghitung metrik...")

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = np.mean(all_ious) if all_ious else 0.0

    # Hitung mAP50
    mAP50 = 0.0
    if all_pred_confs and total_gt > 0:
        sorted_indices = np.argsort(-np.array(all_pred_confs))
        sorted_tp = np.array(all_pred_is_tp)[sorted_indices]

        cum_tp = np.cumsum(sorted_tp)
        cum_fp = np.cumsum(1 - sorted_tp)
        pr_precisions = cum_tp / (cum_tp + cum_fp)
        pr_recalls = cum_tp / total_gt

        mAP50 = compute_ap(pr_recalls.tolist(), pr_precisions.tolist())

    # Hasil
    eval_result = {
        'model_name': model_name,
        'model_path': model_path,
        'model_size_mb': round(model_size, 2),
        'device': device_name,
        'status': 'OK',
        'config': {
            'imgsz': imgsz,
            'conf': conf,
            'iou_threshold': iou_thresh,
        },
        'dataset': {
            'total_images': len(image_files),
            'total_gt_objects': total_gt,
            'total_predictions': len(all_pred_confs),
        },
        'metrics': {
            'precision': round(precision * 100, 2),       # dalam %
            'recall': round(recall * 100, 2),              # dalam %
            'f1_score': round(f1_score * 100, 2),          # dalam %
            'mAP50': round(mAP50 * 100, 2),               # dalam %
            'mean_iou': round(mean_iou * 100, 2),         # dalam %
        },
        'counts': {
            'true_positives': total_tp,
            'false_positives': total_fp,
            'false_negatives': total_fn,
        },
        'system': {
            'hostname': platform.node(),
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        },
    }

    # Simpan JSON
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"eval_accuracy_{model_name}_{device_name}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(eval_result, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"  HASIL EVALUASI AKURASI: {model_name}")
    print(f"{'=' * 70}")
    print(f"  Device              : {device_name}")
    print(f"  Model size          : {model_size:.2f} MB")
    print(f"  Total gambar        : {len(image_files)}")
    print(f"  Total GT objects    : {total_gt}")
    print(f"  Total prediksi      : {len(all_pred_confs)}")
    print(f"")
    print(f"  ── Deteksi ─────────────────────────────────")
    print(f"  True Positives      : {total_tp}")
    print(f"  False Positives     : {total_fp}")
    print(f"  False Negatives     : {total_fn}")
    print(f"")
    print(f"  ── Metrik Akurasi ──────────────────────────")
    print(f"  Precision           : {eval_result['metrics']['precision']:7.2f} %")
    print(f"  Recall              : {eval_result['metrics']['recall']:7.2f} %")
    print(f"  F1-Score            : {eval_result['metrics']['f1_score']:7.2f} %")
    print(f"  mAP@0.5             : {eval_result['metrics']['mAP50']:7.2f} %")
    print(f"  Mean IoU            : {eval_result['metrics']['mean_iou']:7.2f} %")
    print(f"")
    print(f"  📄 {json_path}")
    print(f"{'=' * 70}")

    return eval_result


# ============================================================================
#  MAIN
# ============================================================================

def detect_device_name():
    """Auto-detect nama device berdasarkan platform."""
    node = platform.node().lower()
    plat = platform.platform().lower()

    if 'jetson' in node or 'tegra' in plat:
        return 'jetson_nano'
    elif 'raspberry' in node or 'raspberrypi' in node or ('aarch64' in plat and 'linux' in plat):
        return 'raspi4'
    else:
        return 'pc'


def main():
    parser = argparse.ArgumentParser(
        description="Evaluasi akurasi model YOLOv8 pada edge device"
    )
    parser.add_argument(
        "--model", "-m", type=str, default=None,
        help="Path ke file model (.engine / .tflite / .pt)"
    )
    parser.add_argument(
        "--model-dir", type=str, default=None,
        help="Folder berisi model (evaluasi semua)"
    )
    parser.add_argument(
        "--images", "-i", type=str, required=True,
        help="Path ke folder gambar test"
    )
    parser.add_argument(
        "--labels", "-l", type=str, required=True,
        help="Path ke folder label test (YOLO format .txt)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=SCRIPT_DIR,
        help=f"Folder output JSON (default: {SCRIPT_DIR})"
    )
    parser.add_argument(
        "--conf", type=float, default=DEFAULT_CONF,
        help=f"Confidence threshold (default: {DEFAULT_CONF})"
    )
    parser.add_argument(
        "--iou", type=float, default=DEFAULT_IOU,
        help=f"IoU threshold matching (default: {DEFAULT_IOU})"
    )
    parser.add_argument(
        "--imgsz", type=int, default=DEFAULT_IMGSZ,
        help=f"Ukuran input model (default: {DEFAULT_IMGSZ})"
    )
    parser.add_argument(
        "--device", "-d", type=str, default="cpu",
        help="Device: 'cpu', '0' (GPU)"
    )
    parser.add_argument(
        "--device-name", type=str, default=None,
        help="Nama device (default: auto-detect)"
    )

    args = parser.parse_args()

    device_name = args.device_name or detect_device_name()

    print("=" * 70)
    print("  EVALUASI AKURASI MODEL YOLOv8")
    print(f"  Device: {device_name}")
    print("  Metrik: Precision, Recall, F1-Score, mAP50")
    print("=" * 70)

    all_results = []

    if args.model_dir:
        # Cari semua model
        model_exts = ["*.engine", "*.tflite", "*.pt", "*.onnx"]
        model_files = []
        for ext in model_exts:
            model_files.extend(sorted(glob.glob(os.path.join(args.model_dir, ext))))

        if not model_files:
            print(f"[ERROR] Tidak ada model ditemukan di: {args.model_dir}")
            sys.exit(1)

        print(f"\n[BATCH] Ditemukan {len(model_files)} model:")
        for f in model_files:
            print(f"        - {os.path.basename(f)}")

        for model_file in model_files:
            result = evaluate_model(
                model_path=model_file,
                image_dir=args.images,
                label_dir=args.labels,
                output_dir=args.output,
                conf=args.conf, iou_thresh=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                device_name=device_name,
            )
            if result:
                all_results.append(result)

    elif args.model:
        result = evaluate_model(
            model_path=args.model,
            image_dir=args.images,
            label_dir=args.labels,
            output_dir=args.output,
            conf=args.conf, iou_thresh=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            device_name=device_name,
        )
        if result:
            all_results.append(result)

    else:
        print("[ERROR] Spesifikasikan model: --model path/to/model")
        parser.print_help()
        sys.exit(1)

    # Print summary tabel
    if len(all_results) > 1:
        print(f"\n{'=' * 90}")
        print(f"  RINGKASAN EVALUASI AKURASI — {device_name}")
        print(f"{'=' * 90}")
        print(f"  {'Model':<30s} {'Size(MB)':>10s} {'Precision':>10s} {'Recall':>10s} {'F1':>8s} {'mAP50':>8s}")
        print(f"  {'─' * 30} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 8} {'─' * 8}")
        for r in all_results:
            if r.get('status') == 'OK':
                m = r['metrics']
                print(f"  {r['model_name']:<30s} "
                      f"{r['model_size_mb']:>10.2f} "
                      f"{m['precision']:>9.2f}% "
                      f"{m['recall']:>9.2f}% "
                      f"{m['f1_score']:>7.2f}% "
                      f"{m['mAP50']:>7.2f}%")
            else:
                print(f"  {r['model_name']:<30s} "
                      f"{r.get('model_size_mb', 0):>10.2f} "
                      f"{r.get('status', 'ERROR'):>10s}")
        print(f"{'=' * 90}")

    print("\n[DONE] Evaluasi akurasi selesai!")


if __name__ == "__main__":
    main()
