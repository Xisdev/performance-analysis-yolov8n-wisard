"""
==============================================================================
  Validasi Model YOLOv8 pada Dataset WiSARD IR
==============================================================================
  Jalankan validasi model terhadap val/test split dataset.
  Menghasilkan metrik: mAP@0.5, mAP@0.5:0.95, Precision, Recall.

  Cara pakai:
      python val.py
      python val.py --model models/best_yolov8n.pt --split test
==============================================================================
"""

import os
import argparse
import torch
from ultralytics import YOLO
from multiprocessing import freeze_support

# =====================================================================
#  KONFIGURASI DEFAULT
# =====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

DEFAULT_MODEL = os.path.join(ROOT_DIR, "models", "best_yolov8n.pt")
DEFAULT_DATA = os.path.join(ROOT_DIR, "dataset", "data.yaml")
DEFAULT_PROJECT = os.path.join(ROOT_DIR, "runs", "val")


def jalankan_validasi(model_path, data_path, split, project_dir):
    # --- Deteksi device ---
    if torch.cuda.is_available():
        device_to_use = 0
        print(f"PyTorch   : {torch.__version__}")
        print(f"GPU       : {torch.cuda.get_device_name(0)}")
        print(f"VRAM      : {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
        torch.cuda.empty_cache()
    else:
        device_to_use = "cpu"
        print(f"PyTorch   : {torch.__version__}")
        print("CUDA tidak tersedia. Menggunakan CPU.")

    # --- Validasi keberadaan file ---
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"[ERROR] File model tidak ditemukan:\n  {model_path}")
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"[ERROR] File data.yaml tidak ditemukan:\n  {data_path}")

    run_name = f"yolov8_val_{split}"

    print(f"\n{'=' * 60}")
    print(f"  VALIDASI MODEL YOLOv8 - WiSARD IR")
    print(f"{'=' * 60}")
    print(f"  Model : {model_path}")
    print(f"  Data  : {data_path}")
    print(f"  Split : {split}")
    print(f"  Device: {device_to_use}")
    print(f"  Output: {os.path.join(project_dir, run_name)}")
    print(f"{'=' * 60}\n")

    model = YOLO(model_path)

    results = model.val(
        data=data_path,
        split=split,
        imgsz=640,
        batch=8,
        device=device_to_use,
        workers=2,
        conf=0.001,
        iou=0.6,
        project=project_dir,
        name=run_name,
        exist_ok=True,
        save=True,
        save_json=False,
        save_txt=False,
        save_conf=True,
        plots=True,
        verbose=True,
    )

    # --- Tampilkan ringkasan metrik ---
    print(f"\n{'=' * 60}")
    print(f"  RINGKASAN METRIK VALIDASI")
    print(f"{'=' * 60}")

    box = results.box
    print(f"  mAP@0.5       : {box.map50:.4f}")
    print(f"  mAP@0.5:0.95  : {box.map:.4f}")
    print(f"  Precision (P) : {box.mp:.4f}")
    print(f"  Recall    (R) : {box.mr:.4f}")

    class_names = model.names
    if hasattr(box, "ap_class_index") and box.ap_class_index is not None:
        print(f"\n  Metrik per-kelas (AP@0.5):")
        print(f"  {'Kelas':<20} {'AP@0.5':>10} {'AP@0.5:0.95':>14}")
        print("  " + "-" * 46)
        for i, cls_idx in enumerate(box.ap_class_index):
            cls_name = class_names[int(cls_idx)]
            ap50 = box.ap50[i] if box.ap50 is not None else float("nan")
            ap = box.ap[i] if box.ap is not None else float("nan")
            print(f"  {cls_name:<20} {ap50:>10.4f} {ap:>14.4f}")

    print(f"\n  Hasil validasi disimpan di: {os.path.join(project_dir, run_name)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validasi model YOLOv8 pada WiSARD IR")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Path ke model .pt")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA, help="Path ke data.yaml")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test", "train"],
                        help="Split yang divalidasi (default: val)")
    parser.add_argument("--project", type=str, default=DEFAULT_PROJECT, help="Output directory")
    args = parser.parse_args()

    freeze_support()
    jalankan_validasi(
        model_path=args.model,
        data_path=args.data,
        split=args.split,
        project_dir=args.project,
    )
