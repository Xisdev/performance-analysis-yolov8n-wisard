"""
Generate results.png dari results.csv untuk varian YOLOv8 yang belum punya.
Format identik dengan yang dihasilkan oleh Ultralytics training.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# Konfigurasi
BASE_DIR = r"D:\RISET\drone-wisard\test_for_device\train_result"

# Subplot layout: 2 baris x 5 kolom
# Baris 1: train/box_loss, train/cls_loss, train/dfl_loss, metrics/precision(B), metrics/recall(B)
# Baris 2: val/box_loss, val/cls_loss, val/dfl_loss, metrics/mAP50(B), metrics/mAP50-95(B)
PLOT_COLS = [
    # Row 1
    ['train/box_loss', 'train/cls_loss', 'train/dfl_loss', 
     'metrics/precision(B)', 'metrics/recall(B)'],
    # Row 2
    ['val/box_loss', 'val/cls_loss', 'val/dfl_loss',
     'metrics/mAP50(B)', 'metrics/mAP50-95(B)'],
]

def smooth(y, weight=0.05):
    """Exponential moving average smoothing (sama seperti Ultralytics)."""
    last = y[0]
    smoothed = []
    for v in y:
        smoothed_val = last * weight + (1 - weight) * v
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)


def generate_results_png(csv_path, output_path, model_name):
    """Generate results.png dari results.csv."""
    # Baca CSV
    df = pd.read_csv(csv_path)
    # Strip whitespace dari nama kolom
    df.columns = [c.strip() for c in df.columns]
    
    epochs = df['epoch'].values
    n_epochs = len(epochs)
    
    print(f"\n  Generating: {model_name} ({n_epochs} epochs)")
    
    # Buat figure
    fig, axes = plt.subplots(2, 5, figsize=(16, 6), tight_layout=True)
    
    for row_idx, row_cols in enumerate(PLOT_COLS):
        for col_idx, col_name in enumerate(row_cols):
            ax = axes[row_idx, col_idx]
            
            if col_name not in df.columns:
                ax.set_title(col_name, fontsize=10)
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', 
                       transform=ax.transAxes)
                continue
            
            y = df[col_name].values.astype(float)
            y_smooth = smooth(y)
            
            # Plot data points + line
            ax.plot(epochs, y, marker='.', markersize=3, linewidth=1, 
                   color='#4287f5', label='results', zorder=2)
            # Plot smooth line
            ax.plot(epochs, y_smooth, linewidth=2, linestyle='--',
                   color='#f5a142', label='smooth', zorder=3)
            
            ax.set_title(col_name, fontsize=10)
            
            # Legend hanya di subplot pertama (train/cls_loss)
            if row_idx == 0 and col_idx == 1:
                ax.legend(fontsize=8, loc='upper right')
    
    # X label
    for ax in axes[1]:
        ax.set_xlabel('Epoch', fontsize=8)
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {output_path}")


# Main
print("=" * 60)
print("  Generate results.png untuk varian yang belum ada")
print("=" * 60)

generated = 0
skipped = 0

for folder in sorted(os.listdir(BASE_DIR)):
    folder_path = os.path.join(BASE_DIR, folder)
    if not os.path.isdir(folder_path):
        continue
    
    csv_path = os.path.join(folder_path, "results.csv")
    png_path = os.path.join(folder_path, "results.png")
    
    if not os.path.exists(csv_path):
        print(f"  [SKIP] {folder}: no results.csv")
        continue
    
    if os.path.exists(png_path):
        print(f"  [OVERWRITE] {folder}: regenerating results.png")
    
    generate_results_png(csv_path, png_path, folder)
    generated += 1

print(f"\n{'=' * 60}")
print(f"  Generated: {generated}, Skipped (already exists): {skipped}")
print(f"{'=' * 60}")
