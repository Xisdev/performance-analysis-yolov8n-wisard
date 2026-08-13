"""
=============================================================================
  YOLOv8n Model Evaluation Script
  Dataset  : AIResQ Benchmark (Infrared Human Detection)
  Model    : best.pt (yolov8n_optimized)
  Output   : D:/RISET/drone-wisard/test_performa/
=============================================================================
"""

import os
import sys
import json
import time
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from collections import defaultdict
from multiprocessing import freeze_support

# ── Paths ──────────────────────────────────────────────────────────────────
MODEL_PATH   = r"D:\RISET\drone-wisard\runs_yolov8n_wisard\runs_yolov8n_wisard\yolov8n_optimized\weights\best.pt"
DATA_YAML    = r"D:\RISET\drone-wisard\datasets\AIResQ\Benchmark\data.yaml"
TEST_IMG_DIR = r"D:\RISET\drone-wisard\datasets\AIResQ\Benchmark\images"
TEST_LBL_DIR = r"D:\RISET\drone-wisard\datasets\AIResQ\Benchmark\labels"
OUTPUT_DIR   = r"D:\RISET\drone-wisard\test_performa\AIResQ\AIResQ_Output"

# Clean up previous output directory so we can run fresh
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────
CONF_THRESHOLD = 0.25          # Default confidence threshold
IOU_THRESHOLD  = 0.5           # IoU threshold for TP/FP matching
IMG_SIZE       = 640           # Inference image size

CLASS_NAMES = {0: "human"}

# ── Utility Functions ──────────────────────────────────────────────────────

def xywh_to_xyxy(box):
    """Convert YOLO xywh (normalized) to xyxy (normalized)."""
    cx, cy, w, h = box
    return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]


def compute_iou(box1, box2):
    """Compute IoU between two boxes in xyxy format."""
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
    """Load ground truth labels from YOLO txt file."""
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


def match_predictions(gt_boxes, pred_boxes, iou_thresh=0.5):
    """
    Match predictions to ground truths using IoU.
    Returns TP, FP, FN counts and matched details.
    """
    gt_matched = [False] * len(gt_boxes)
    pred_matched = [False] * len(pred_boxes)

    # Sort predictions by confidence (descending)
    sorted_preds = sorted(enumerate(pred_boxes), key=lambda x: x[1]['conf'], reverse=True)

    tp_details = []
    fp_details = []

    for pred_idx, pred in sorted_preds:
        pred_xyxy = xywh_to_xyxy(pred['bbox'])
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(gt_boxes):
            if gt_matched[gt_idx]:
                continue
            if gt['class'] != pred['class']:
                continue
            gt_xyxy = xywh_to_xyxy(gt['bbox'])
            iou = compute_iou(pred_xyxy, gt_xyxy)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_thresh and best_gt_idx >= 0:
            gt_matched[best_gt_idx] = True
            pred_matched[pred_idx] = True
            tp_details.append({'pred': pred, 'iou': best_iou})
        else:
            fp_details.append({'pred': pred, 'best_iou': best_iou})

    fn_count = sum(1 for m in gt_matched if not m)
    tp_count = len(tp_details)
    fp_count = len(fp_details)

    return tp_count, fp_count, fn_count, tp_details, fp_details


# ── Visualization Style ───────────────────────────────────────────────────

def setup_plot_style():
    """Configure premium matplotlib style."""
    plt.rcParams.update({
        'figure.facecolor': '#0d1117',
        'axes.facecolor': '#161b22',
        'axes.edgecolor': '#30363d',
        'axes.labelcolor': '#c9d1d9',
        'axes.titlepad': 15,
        'text.color': '#c9d1d9',
        'xtick.color': '#8b949e',
        'ytick.color': '#8b949e',
        'grid.color': '#21262d',
        'grid.alpha': 0.6,
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'figure.dpi': 150,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
        'savefig.facecolor': '#0d1117',
        'savefig.edgecolor': '#0d1117',
    })


# Color palette
COLORS = {
    'primary':    '#58a6ff',
    'success':    '#3fb950',
    'warning':    '#d29922',
    'danger':     '#f85149',
    'purple':     '#bc8cff',
    'cyan':       '#39d2c0',
    'orange':     '#f0883e',
    'pink':       '#f778ba',
    'gradient':   ['#58a6ff', '#bc8cff', '#f778ba', '#f85149', '#d29922', '#3fb950'],
}


# ── Main Evaluation ───────────────────────────────────────────────────────

def main():
    setup_plot_style()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  YOLOv8n Model Evaluation — AIResQ Benchmark Dataset")
    print("=" * 70)

    # ── Step 1: Load model & run ultralytics val() ─────────────────────
    import torch
    from ultralytics import YOLO

    print(f"\n📦 Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    device_name = "CPU"
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
    print(f"🖥️  Device: {device_name}")

    print(f"\n🔍 Running official validation on test set...")
    t_start = time.time()
    val_results = model.val(
        data=DATA_YAML,
        split='test',
        imgsz=IMG_SIZE,
        batch=8,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        device=0 if torch.cuda.is_available() else 'cpu',
        save_json=True,
        plots=True,          # This generates Confusion Matrix, PR curve, etc.
        project=OUTPUT_DIR,
        name='ultralytics_val',
        exist_ok=True,
        verbose=False,
    )
    val_time = time.time() - t_start
    print(f"   ✅ Validation completed in {val_time:.1f}s")
    print(f"   📁 Confusion Matrix & Ultralytics plots saved to: {os.path.join(OUTPUT_DIR, 'ultralytics_val')}")

    # Extract official metrics & Test Loss
    d = val_results.results_dict
    official_metrics = {
        'mAP50':       float(val_results.box.map50),
        'mAP50-95':    float(val_results.box.map),
        'Precision':   float(val_results.box.mp),
        'Recall':      float(val_results.box.mr),
        'Box_Loss':    float(d.get('val/box_loss', 0.0)),
        'Class_Loss':  float(d.get('val/cls_loss', 0.0)),
        'DFL_Loss':    float(d.get('val/dfl_loss', 0.0)),
    }
    # F1 from official P and R
    p_off, r_off = official_metrics['Precision'], official_metrics['Recall']
    official_metrics['F1'] = 2 * p_off * r_off / (p_off + r_off) if (p_off + r_off) > 0 else 0.0

    print(f"\n📊 Official Ultralytics Metrics:")
    for k, v in official_metrics.items():
        print(f"   {k:>12s}: {v:.4f}")

    # ── Step 2: Custom per-image evaluation ────────────────────────────
    print(f"\n🔬 Running custom per-image evaluation...")

    image_files = sorted([
        f for f in os.listdir(TEST_IMG_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ])
    total_images = len(image_files)
    print(f"   Found {total_images} test images")

    # Storage for per-image results
    all_tp, all_fp, all_fn = 0, 0, 0
    per_image_results = []
    all_confidences = []
    all_ious = []
    all_pred_is_tp = []  # Whether each prediction is TP (for PR curve)
    all_pred_confs = []  # Confidence of each prediction
    total_gt_count = 0
    all_speeds = []      # For FPS calculation

    # Directory to save visual prediction results
    PRED_VIS_DIR = os.path.join(OUTPUT_DIR, "predictions_visualized")
    os.makedirs(PRED_VIS_DIR, exist_ok=True)
    print(f"   📁 Prediction images will be saved to: {PRED_VIS_DIR}")

    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(TEST_IMG_DIR, img_file)
        lbl_file = os.path.splitext(img_file)[0] + '.txt'
        lbl_path = os.path.join(TEST_LBL_DIR, lbl_file)

        # Ground truth
        gt_labels = load_gt_labels(lbl_path)
        total_gt_count += len(gt_labels)

        # Prediction
        results = model.predict(
            img_path,
            imgsz=IMG_SIZE,
            conf=CONF_THRESHOLD,
            iou=0.45,  # NMS IoU
            device=0 if torch.cuda.is_available() else 'cpu',
            verbose=False,
        )

        # Parse predictions
        pred_boxes = []
        if len(results) > 0:
            res = results[0]
            
            # Save visual prediction
            res.save(filename=os.path.join(PRED_VIS_DIR, img_file))
            
            # Record speed for FPS (preprocess + inference + postprocess) in ms
            if hasattr(res, 'speed') and res.speed is not None:
                all_speeds.append(sum(res.speed.values()))
                
            if res.boxes is not None:
                boxes = res.boxes
                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls  = int(boxes.cls[i].cpu().numpy())
                    # Convert xyxy to normalized xywh
                    img_h, img_w = results[0].orig_shape
                    cx = ((xyxy[0] + xyxy[2]) / 2) / img_w
                    cy = ((xyxy[1] + xyxy[3]) / 2) / img_h
                    bw = (xyxy[2] - xyxy[0]) / img_w
                    bh = (xyxy[3] - xyxy[1]) / img_h
                    pred_boxes.append({
                        'class': cls,
                        'bbox': [cx, cy, bw, bh],
                        'conf': conf,
                    })
                    all_confidences.append(conf)

        # Match
        tp, fp, fn, tp_details, fp_details = match_predictions(
            gt_labels, pred_boxes, IOU_THRESHOLD
        )
        all_tp += tp
        all_fp += fp
        all_fn += fn

        # Collect IoU values from TP
        for d in tp_details:
            all_ious.append(d['iou'])

        # For PR curve: record each prediction as TP or FP
        sorted_preds = sorted(pred_boxes, key=lambda x: x['conf'], reverse=True)
        gt_matched_pr = [False] * len(gt_labels)
        for pred in sorted_preds:
            pred_xyxy = xywh_to_xyxy(pred['bbox'])
            best_iou = 0
            best_gt_idx = -1
            for gt_idx, gt in enumerate(gt_labels):
                if gt_matched_pr[gt_idx]:
                    continue
                gt_xyxy = xywh_to_xyxy(gt['bbox'])
                iou = compute_iou(pred_xyxy, gt_xyxy)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            if best_iou >= IOU_THRESHOLD and best_gt_idx >= 0:
                gt_matched_pr[best_gt_idx] = True
                all_pred_is_tp.append(1)
            else:
                all_pred_is_tp.append(0)
            all_pred_confs.append(pred['conf'])

        img_precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        img_recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        img_f1 = 2 * img_precision * img_recall / (img_precision + img_recall) if (img_precision + img_recall) > 0 else 0.0

        per_image_results.append({
            'image': img_file,
            'gt_count': len(gt_labels),
            'pred_count': len(pred_boxes),
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': img_precision,
            'recall': img_recall,
            'f1': img_f1,
        })

        if (idx + 1) % 100 == 0 or (idx + 1) == total_images:
            print(f"   Processed {idx+1}/{total_images} images...")

    # ── Step 3: Compute aggregate metrics ──────────────────────────────
    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    recall    = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    f1_score  = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = all_tp / (all_tp + all_fp + all_fn) if (all_tp + all_fp + all_fn) > 0 else 0.0

    mean_iou = np.mean(all_ious) if all_ious else 0.0
    mean_conf = np.mean(all_confidences) if all_confidences else 0.0

    avg_inf_time = np.mean(all_speeds) if all_speeds else 0.0
    fps = 1000.0 / avg_inf_time if avg_inf_time > 0 else 0.0

    # Compute PR curve data
    sorted_indices = np.argsort(-np.array(all_pred_confs))
    sorted_tp = np.array(all_pred_is_tp)[sorted_indices]
    sorted_confs = np.array(all_pred_confs)[sorted_indices]

    cum_tp = np.cumsum(sorted_tp)
    cum_fp = np.cumsum(1 - sorted_tp)
    pr_precisions = cum_tp / (cum_tp + cum_fp)
    pr_recalls = cum_tp / total_gt_count if total_gt_count > 0 else cum_tp

    # Compute F1 at different confidence thresholds
    conf_thresholds = np.arange(0.1, 0.95, 0.05)
    f1_vs_conf = []
    prec_vs_conf = []
    rec_vs_conf = []

    for ct in conf_thresholds:
        ct_tp, ct_fp, ct_fn = 0, 0, 0
        for res in per_image_results:
            # Re-count using confidence filter (approximate from stored data)
            pass  # We'll compute differently below

    # Recompute metrics at different thresholds using all predictions
    f1_vs_conf = []
    prec_vs_conf = []
    rec_vs_conf = []
    for ct in conf_thresholds:
        mask = sorted_confs >= ct
        tp_at_ct = sorted_tp[mask].sum() if mask.any() else 0
        fp_at_ct = (1 - sorted_tp[mask]).sum() if mask.any() else 0
        fn_at_ct = total_gt_count - tp_at_ct
        p_ct = tp_at_ct / (tp_at_ct + fp_at_ct) if (tp_at_ct + fp_at_ct) > 0 else 0
        r_ct = tp_at_ct / total_gt_count if total_gt_count > 0 else 0
        f1_ct = 2 * p_ct * r_ct / (p_ct + r_ct) if (p_ct + r_ct) > 0 else 0
        prec_vs_conf.append(p_ct)
        rec_vs_conf.append(r_ct)
        f1_vs_conf.append(f1_ct)

    # Best F1 threshold
    best_f1_idx = np.argmax(f1_vs_conf)
    best_f1_conf = conf_thresholds[best_f1_idx]
    best_f1_val = f1_vs_conf[best_f1_idx]

    # ── Print Results ──────────────────────────────────────────────────
    custom_metrics = {
        'Total Images':         total_images,
        'Total GT Objects':     total_gt_count,
        'Total Predictions':    len(all_confidences),
        'True Positives (TP)':  all_tp,
        'False Positives (FP)': all_fp,
        'False Negatives (FN)': all_fn,
        'Precision':            precision,
        'Recall':               recall,
        'F1 Score':             f1_score,
        'Accuracy (Jaccard)':   accuracy,
        'Mean IoU (TP)':        mean_iou,
        'Mean Confidence':      mean_conf,
        'Best F1 Threshold':    best_f1_conf,
        'Best F1 Score':        best_f1_val,
        'Avg Inference (ms)':   avg_inf_time,
        'Speed (FPS)':          fps,
    }

    print(f"\n{'='*70}")
    print(f"  CUSTOM EVALUATION RESULTS (IoU Thresh = {IOU_THRESHOLD})")
    print(f"{'='*70}")
    for k, v in custom_metrics.items():
        if isinstance(v, float):
            print(f"   {k:>25s}: {v:.4f}")
        else:
            print(f"   {k:>25s}: {v}")

    # ── Step 4: Save results to JSON ───────────────────────────────────
    results_data = {
        'model_path': MODEL_PATH,
        'dataset': DATA_YAML,
        'test_images_dir': TEST_IMG_DIR,
        'image_size': IMG_SIZE,
        'confidence_threshold': CONF_THRESHOLD,
        'iou_threshold': IOU_THRESHOLD,
        'official_metrics': official_metrics,
        'custom_metrics': {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in custom_metrics.items()},
        'best_f1_threshold': float(best_f1_conf),
        'best_f1_value': float(best_f1_val),
    }

    json_path = os.path.join(OUTPUT_DIR, 'test_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Results saved to: {json_path}")

    # ── Step 5: Generate Visualizations ────────────────────────────────
    print(f"\n🎨 Generating visualizations...")

    # ═══════════════════════════════════════════════════════════════════
    # FIGURE 1: Metrics Summary Dashboard
    # ═══════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('YOLOv8n Model Evaluation — AIResQ Benchmark Dataset',
                 fontsize=18, fontweight='bold', color='white', y=0.98)

    # 1a: Bar chart — Key Metrics
    ax = axes[0, 0]
    metrics_names = ['Precision', 'Recall', 'F1 Score', 'mAP50', 'mAP50-95']
    metrics_vals = [precision, recall, f1_score,
                    official_metrics['mAP50'], official_metrics['mAP50-95']]
    bar_colors = [COLORS['primary'], COLORS['success'], COLORS['warning'],
                  COLORS['purple'], COLORS['cyan']]
    bars = ax.bar(metrics_names, metrics_vals, color=bar_colors, width=0.6,
                  edgecolor='none', alpha=0.9)
    for bar, val in zip(bars, metrics_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold',
                fontsize=12, color='white')
    ax.set_ylim(0, 1.15)
    ax.set_title('Key Performance Metrics', fontweight='bold', color='white')
    ax.set_ylabel('Score')
    ax.grid(axis='y', alpha=0.3)

    # 1b: Stacked bar — TP / FP / FN
    ax = axes[0, 1]
    categories = ['True Positives', 'False Positives', 'False Negatives']
    values = [all_tp, all_fp, all_fn]
    colors_bar = [COLORS['success'], COLORS['danger'], COLORS['warning']]
    bars = ax.barh(categories, values, color=colors_bar, height=0.5, edgecolor='none')
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values)*0.02, bar.get_y() + bar.get_height()/2,
                f'{val}', ha='left', va='center', fontweight='bold',
                fontsize=13, color='white')
    ax.set_title('Detection Breakdown', fontweight='bold', color='white')
    ax.set_xlabel('Count')
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()

    # 1c: Confidence Distribution
    ax = axes[1, 0]
    if all_confidences:
        ax.hist(all_confidences, bins=50, color=COLORS['primary'], alpha=0.7,
                edgecolor='#30363d', linewidth=0.5)
        ax.axvline(mean_conf, color=COLORS['danger'], linestyle='--', linewidth=2,
                   label=f'Mean: {mean_conf:.3f}')
        ax.axvline(CONF_THRESHOLD, color=COLORS['warning'], linestyle='--', linewidth=2,
                   label=f'Threshold: {CONF_THRESHOLD}')
    ax.set_title('Confidence Score Distribution', fontweight='bold', color='white')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Frequency')
    ax.legend(facecolor='#161b22', edgecolor='#30363d', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # 1d: IoU Distribution (TP only)
    ax = axes[1, 1]
    if all_ious:
        ax.hist(all_ious, bins=40, color=COLORS['success'], alpha=0.7,
                edgecolor='#30363d', linewidth=0.5, range=(0.5, 1.0))
        ax.axvline(mean_iou, color=COLORS['danger'], linestyle='--', linewidth=2,
                   label=f'Mean IoU: {mean_iou:.3f}')
    ax.set_title('IoU Distribution (True Positives)', fontweight='bold', color='white')
    ax.set_xlabel('IoU')
    ax.set_ylabel('Frequency')
    ax.legend(facecolor='#161b22', edgecolor='#30363d', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUTPUT_DIR, '01_metrics_dashboard.png'))
    plt.close(fig)
    print("   ✅ 01_metrics_dashboard.png")

    # ═══════════════════════════════════════════════════════════════════
    # FIGURE 2: Precision-Recall Curve
    # ═══════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(10, 8))
    if len(pr_precisions) > 0:
        # Smooth the PR curve
        recall_interp = np.linspace(0, 1, 1000)
        # Ensure monotonically decreasing precision
        pr_prec_smooth = np.copy(pr_precisions)
        for i in range(len(pr_prec_smooth) - 2, -1, -1):
            pr_prec_smooth[i] = max(pr_prec_smooth[i], pr_prec_smooth[i+1])

        ax.fill_between(pr_recalls, pr_prec_smooth, alpha=0.15, color=COLORS['primary'])
        ax.plot(pr_recalls, pr_prec_smooth, color=COLORS['primary'], linewidth=2.5,
                label=f'PR Curve (mAP@0.5 = {official_metrics["mAP50"]:.3f})')

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Recall', fontsize=13)
    ax.set_ylabel('Precision', fontsize=13)
    ax.set_title('Precision-Recall Curve', fontsize=16, fontweight='bold', color='white')
    ax.legend(loc='lower left', facecolor='#161b22', edgecolor='#30363d', fontsize=12)
    ax.grid(True, alpha=0.3)

    fig.savefig(os.path.join(OUTPUT_DIR, '02_precision_recall_curve.png'))
    plt.close(fig)
    print("   ✅ 02_precision_recall_curve.png")

    # ═══════════════════════════════════════════════════════════════════
    # FIGURE 3: F1 vs Confidence Threshold
    # ═══════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(conf_thresholds, f1_vs_conf, color=COLORS['warning'], linewidth=2.5,
            marker='o', markersize=6, label='F1 Score')
    ax.plot(conf_thresholds, prec_vs_conf, color=COLORS['primary'], linewidth=2,
            linestyle='--', marker='s', markersize=4, label='Precision')
    ax.plot(conf_thresholds, rec_vs_conf, color=COLORS['success'], linewidth=2,
            linestyle='--', marker='^', markersize=4, label='Recall')

    ax.axvline(best_f1_conf, color=COLORS['danger'], linestyle=':', linewidth=2,
               label=f'Best F1 @ conf={best_f1_conf:.2f} ({best_f1_val:.3f})')
    ax.scatter([best_f1_conf], [best_f1_val], color=COLORS['danger'], s=120,
               zorder=5, edgecolors='white', linewidth=2)

    ax.set_xlabel('Confidence Threshold', fontsize=13)
    ax.set_ylabel('Score', fontsize=13)
    ax.set_title('F1 / Precision / Recall vs Confidence Threshold',
                 fontsize=16, fontweight='bold', color='white')
    ax.legend(facecolor='#161b22', edgecolor='#30363d', fontsize=11)
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    fig.savefig(os.path.join(OUTPUT_DIR, '03_f1_vs_confidence.png'))
    plt.close(fig)
    print("   ✅ 03_f1_vs_confidence.png")

    # ═══════════════════════════════════════════════════════════════════
    # FIGURE 4: Per-Image Detection Analysis
    # ═══════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    img_precisions = [r['precision'] for r in per_image_results]
    img_recalls    = [r['recall'] for r in per_image_results]
    img_f1s        = [r['f1'] for r in per_image_results]

    # 4a: Per-image precision histogram
    ax = axes[0]
    ax.hist(img_precisions, bins=30, color=COLORS['primary'], alpha=0.7,
            edgecolor='#30363d', linewidth=0.5)
    ax.axvline(np.mean(img_precisions), color=COLORS['danger'], linestyle='--',
               linewidth=2, label=f'Mean: {np.mean(img_precisions):.3f}')
    ax.set_title('Per-Image Precision', fontweight='bold', color='white')
    ax.set_xlabel('Precision')
    ax.set_ylabel('Image Count')
    ax.legend(facecolor='#161b22', edgecolor='#30363d')
    ax.grid(axis='y', alpha=0.3)

    # 4b: Per-image recall histogram
    ax = axes[1]
    ax.hist(img_recalls, bins=30, color=COLORS['success'], alpha=0.7,
            edgecolor='#30363d', linewidth=0.5)
    ax.axvline(np.mean(img_recalls), color=COLORS['danger'], linestyle='--',
               linewidth=2, label=f'Mean: {np.mean(img_recalls):.3f}')
    ax.set_title('Per-Image Recall', fontweight='bold', color='white')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Image Count')
    ax.legend(facecolor='#161b22', edgecolor='#30363d')
    ax.grid(axis='y', alpha=0.3)

    # 4c: Per-image F1 histogram
    ax = axes[2]
    ax.hist(img_f1s, bins=30, color=COLORS['warning'], alpha=0.7,
            edgecolor='#30363d', linewidth=0.5)
    ax.axvline(np.mean(img_f1s), color=COLORS['danger'], linestyle='--',
               linewidth=2, label=f'Mean: {np.mean(img_f1s):.3f}')
    ax.set_title('Per-Image F1 Score', fontweight='bold', color='white')
    ax.set_xlabel('F1 Score')
    ax.set_ylabel('Image Count')
    ax.legend(facecolor='#161b22', edgecolor='#30363d')
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Per-Image Detection Quality Distribution',
                 fontsize=16, fontweight='bold', color='white', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '04_per_image_analysis.png'))
    plt.close(fig)
    print("   ✅ 04_per_image_analysis.png")

    # ═══════════════════════════════════════════════════════════════════
    # FIGURE 5: GT vs Predicted Object Count Scatter
    # ═══════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(10, 8))
    gt_counts  = [r['gt_count'] for r in per_image_results]
    pred_counts = [r['pred_count'] for r in per_image_results]
    f1_colors = [r['f1'] for r in per_image_results]

    scatter = ax.scatter(gt_counts, pred_counts, c=f1_colors, cmap='RdYlGn',
                         s=40, alpha=0.7, edgecolors='#30363d', linewidth=0.5,
                         vmin=0, vmax=1)
    max_count = max(max(gt_counts, default=1), max(pred_counts, default=1)) + 1
    ax.plot([0, max_count], [0, max_count], color=COLORS['danger'], linestyle='--',
            linewidth=1.5, alpha=0.7, label='Perfect Match')
    ax.set_xlabel('Ground Truth Count', fontsize=13)
    ax.set_ylabel('Predicted Count', fontsize=13)
    ax.set_title('GT vs Predicted Object Count per Image',
                 fontsize=16, fontweight='bold', color='white')
    ax.legend(facecolor='#161b22', edgecolor='#30363d', fontsize=11)
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('F1 Score', color='#c9d1d9')
    cbar.ax.yaxis.set_tick_params(color='#8b949e')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#8b949e')
    ax.grid(True, alpha=0.3)

    fig.savefig(os.path.join(OUTPUT_DIR, '05_gt_vs_pred_count.png'))
    plt.close(fig)
    print("   ✅ 05_gt_vs_pred_count.png")

    # ═══════════════════════════════════════════════════════════════════
    # FIGURE 6: Confusion-like Pie Chart
    # ═══════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # 6a: Pie — detection outcome
    ax = axes[0]
    pie_labels = ['True Positives', 'False Positives', 'False Negatives']
    pie_values = [all_tp, all_fp, all_fn]
    pie_colors = [COLORS['success'], COLORS['danger'], COLORS['warning']]
    explode = (0.05, 0.05, 0.05)
    wedges, texts, autotexts = ax.pie(
        pie_values, labels=pie_labels, colors=pie_colors, autopct='%1.1f%%',
        startangle=90, explode=explode, textprops={'color': 'white', 'fontsize': 12},
        pctdistance=0.75, wedgeprops={'edgecolor': '#0d1117', 'linewidth': 2}
    )
    for t in autotexts:
        t.set_fontweight('bold')
    ax.set_title('Detection Outcome Distribution', fontweight='bold',
                 color='white', fontsize=14)

    # 6b: Donut — prediction quality
    ax = axes[1]
    perfect = sum(1 for r in per_image_results if r['f1'] == 1.0)
    good = sum(1 for r in per_image_results if 0.5 <= r['f1'] < 1.0)
    poor = sum(1 for r in per_image_results if 0 < r['f1'] < 0.5)
    missed = sum(1 for r in per_image_results if r['f1'] == 0.0)

    donut_labels = ['Perfect (F1=1.0)', 'Good (F1≥0.5)', 'Poor (F1<0.5)', 'Missed (F1=0)']
    donut_values = [perfect, good, poor, missed]
    donut_colors = [COLORS['success'], COLORS['primary'], COLORS['warning'], COLORS['danger']]

    # Filter out zero values
    non_zero = [(l, v, c) for l, v, c in zip(donut_labels, donut_values, donut_colors) if v > 0]
    if non_zero:
        d_labels, d_values, d_colors = zip(*non_zero)
        wedges2, texts2, autotexts2 = ax.pie(
            d_values, labels=d_labels, colors=d_colors, autopct='%1.1f%%',
            startangle=90, textprops={'color': 'white', 'fontsize': 11},
            pctdistance=0.8, wedgeprops={'edgecolor': '#0d1117', 'linewidth': 2, 'width': 0.5}
        )
        for t in autotexts2:
            t.set_fontweight('bold')
    ax.set_title('Image-Level Detection Quality', fontweight='bold',
                 color='white', fontsize=14)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '06_detection_distribution.png'))
    plt.close(fig)
    print("   ✅ 06_detection_distribution.png")

    # ═══════════════════════════════════════════════════════════════════
    # FIGURE 7: Rolling F1 across images (temporal trend)
    # ═══════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(14, 6))
    window = min(20, total_images // 5) if total_images > 20 else 5
    f1_series = np.array(img_f1s)
    if len(f1_series) > window:
        rolling_f1 = np.convolve(f1_series, np.ones(window)/window, mode='valid')
        x_rolling = range(window - 1, len(f1_series))
        ax.plot(range(len(f1_series)), f1_series, alpha=0.25, color=COLORS['primary'],
                linewidth=0.8, label='Per-Image F1')
        ax.plot(x_rolling, rolling_f1, color=COLORS['warning'], linewidth=2.5,
                label=f'Rolling Avg (window={window})')
    else:
        ax.plot(range(len(f1_series)), f1_series, color=COLORS['primary'], linewidth=1.5,
                label='Per-Image F1')

    ax.axhline(np.mean(f1_series), color=COLORS['danger'], linestyle='--',
               linewidth=1.5, alpha=0.8, label=f'Overall Mean: {np.mean(f1_series):.3f}')
    ax.set_xlabel('Image Index', fontsize=13)
    ax.set_ylabel('F1 Score', fontsize=13)
    ax.set_title('F1 Score Trend Across Test Images',
                 fontsize=16, fontweight='bold', color='white')
    ax.legend(facecolor='#161b22', edgecolor='#30363d', fontsize=11)
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)

    fig.savefig(os.path.join(OUTPUT_DIR, '07_f1_trend.png'))
    plt.close(fig)
    print("   ✅ 07_f1_trend.png")

    # ═══════════════════════════════════════════════════════════════════
    # FIGURE 8: Summary Report Card
    # ═══════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.axis('off')

    # Title
    ax.text(0.5, 0.95, 'YOLOv8n — AIResQ Benchmark Test Evaluation Report',
            transform=ax.transAxes, fontsize=22, fontweight='bold',
            ha='center', va='top', color='white',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#1f6feb', alpha=0.8))

    # Metrics table
    table_data = [
        ['Metric', 'Value'],
        ['Total Test Images', f'{total_images}'],
        ['Total GT Objects', f'{total_gt_count}'],
        ['Total Predictions', f'{len(all_confidences)}'],
        ['True Positives (TP)', f'{all_tp}'],
        ['False Positives (FP)', f'{all_fp}'],
        ['False Negatives (FN)', f'{all_fn}'],
        ['─' * 25, '─' * 15],
        ['mAP@0.5', f'{official_metrics["mAP50"]:.4f}'],
        ['mAP@0.5:0.95', f'{official_metrics["mAP50-95"]:.4f}'],
        ['Box Loss', f'{official_metrics["Box_Loss"]:.4f}'],
        ['Class Loss', f'{official_metrics["Class_Loss"]:.4f}'],
        ['DFL Loss', f'{official_metrics["DFL_Loss"]:.4f}'],
        ['─' * 25, '─' * 15],
        ['Precision', f'{precision:.4f}'],
        ['Recall', f'{recall:.4f}'],
        ['F1 Score', f'{f1_score:.4f}'],
        ['Accuracy (Jaccard)', f'{accuracy:.4f}'],
        ['Mean IoU (TP)', f'{mean_iou:.4f}'],
        ['Mean Confidence', f'{mean_conf:.4f}'],
        ['Best F1 Threshold', f'{best_f1_conf:.2f} (F1={best_f1_val:.4f})'],
        ['Inference Time', f'{avg_inf_time:.2f} ms / img'],
        ['Speed', f'{fps:.2f} FPS'],
    ]

    table = ax.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc='center',
        cellLoc='center',
        colWidths=[0.45, 0.3],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)

    # Style the table
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#30363d')
        if row == 0:
            cell.set_facecolor('#1f6feb')
            cell.set_text_props(color='white', fontweight='bold', fontsize=13)
        elif '─' in cell.get_text().get_text():
            cell.set_facecolor('#21262d')
            cell.set_text_props(color='#30363d')
        else:
            cell.set_facecolor('#161b22')
            cell.set_text_props(color='#c9d1d9')

    fig.savefig(os.path.join(OUTPUT_DIR, '08_summary_report.png'))
    plt.close(fig)
    print("   ✅ 08_summary_report.png")

    # ── Save per-image CSV ─────────────────────────────────────────────
    csv_path = os.path.join(OUTPUT_DIR, 'per_image_results.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('image,gt_count,pred_count,tp,fp,fn,precision,recall,f1\n')
        for r in per_image_results:
            f.write(f"{r['image']},{r['gt_count']},{r['pred_count']},"
                    f"{r['tp']},{r['fp']},{r['fn']},"
                    f"{r['precision']:.4f},{r['recall']:.4f},{r['f1']:.4f}\n")
    print(f"   💾 Per-image CSV: {csv_path}")

    print(f"\n{'='*70}")
    print(f"  ✅ ALL DONE! Results saved in: {OUTPUT_DIR}")
    print(f"{'='*70}")
    print(f"\nGenerated files:")
    print(f"   📄 test_results.json        — Full metrics in JSON")
    print(f"   📄 per_image_results.csv     — Per-image breakdown")
    print(f"   📊 01_metrics_dashboard.png  — Key metrics bar chart")
    print(f"   📊 02_precision_recall_curve — PR curve")
    print(f"   📊 03_f1_vs_confidence.png   — F1/P/R vs threshold")
    print(f"   📊 04_per_image_analysis.png — Per-image distributions")
    print(f"   📊 05_gt_vs_pred_count.png   — GT vs Pred scatter")
    print(f"   📊 06_detection_distribution — TP/FP/FN pie charts")
    print(f"   📊 07_f1_trend.png           — F1 trend over images")
    print(f"   📊 08_summary_report.png     — Full report card")
    print(f"   📁 ultralytics_val/          — Contains Confusion Matrix & PR plots")
    print(f"   📁 predictions_visualized/   — Images with predicted bounding boxes")


if __name__ == '__main__':
    freeze_support()
    main()
