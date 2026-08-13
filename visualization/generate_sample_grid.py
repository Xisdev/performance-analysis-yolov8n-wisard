"""
Generate Grid Sampel Deteksi 2x2
Mengambil 4 gambar acak dari dataset, deteksi dengan model, lalu gabung jadi grid.

Cara pakai:
    python generate_sample_grid.py
    python generate_sample_grid.py --model models/best_yolov8n.pt
"""
import os
import random
import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# Definisi Path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

DEFAULT_DATASET = os.path.join(ROOT_DIR, "dataset")
DEFAULT_MODEL = os.path.join(ROOT_DIR, "models", "best_yolov8n.pt")
DEFAULT_OUTPUT = os.path.join(ROOT_DIR, "results", "charts", "sample_detection_grid.jpg")


def main():
    parser = argparse.ArgumentParser(description="Generate grid sampel deteksi 2x2")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Path ke model .pt")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="Path ke folder dataset")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Path output gambar")
    args = parser.parse_args()

    print(f"Memuat model dari: {args.model}")
    model = YOLO(args.model)

    print(f"Mencari gambar di dataset: {args.dataset}")
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    all_images = []
    for ext in image_extensions:
        all_images.extend(list(Path(args.dataset).rglob(ext)))

    if len(all_images) < 4:
        print(f"Gambar tidak cukup! Hanya ditemukan {len(all_images)} gambar, butuh minimal 4.")
        return

    selected_images = random.sample(all_images, 4)
    print("\n4 Gambar yang terpilih:")
    for img_path in selected_images:
        print(f" - {img_path}")

    plotted_images = []
    target_size = (640, 640)

    for img_path in selected_images:
        print(f"Memproses: {img_path.name}...")
        results = model(str(img_path), device='cpu')
        res_img = results[0].plot()
        resized_img = cv2.resize(res_img, target_size)
        plotted_images.append(resized_img)

    top_row = np.hstack((plotted_images[0], plotted_images[1]))
    bottom_row = np.hstack((plotted_images[2], plotted_images[3]))
    grid_image = np.vstack((top_row, bottom_row))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cv2.imwrite(args.output, grid_image)
    print(f"\nBerhasil! Gambar grid disimpan di: {args.output}")


if __name__ == "__main__":
    main()
