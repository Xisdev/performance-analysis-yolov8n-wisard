"""
Grafik Perbandingan mAP50 Antar Platform pada Setiap Varian Model YOLOv8
"""
import matplotlib.pyplot as plt
import numpy as np
import os

# Output directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(ROOT_DIR, "results", "charts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data mAP50 dari Tabel 4.5 - Evaluasi Akurasi Konversi Model
models = ['YOLOv8n', 'YOLOv8s', 'YOLOv8m', 'YOLOv8l', 'YOLOv8x']

# Nilai mAP50 (%) untuk setiap platform
baseline_pc = [53.20, 51.30, 49.48, 57.92, 53.63]       # Baseline PC (.pt)
jetson_nano = [51.82, 43.23, 48.23, 41.00, 42.54]        # Jetson Nano (engine FP16)
raspberry_pi = [51.59, 43.27, 48.23, 40.94, 41.94]       # Raspberry Pi 4 (tflite FP32)

x = np.arange(len(models))
bar_width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))

colors = {
    'baseline': '#4472C4',
    'jetson': '#ED7D31',
    'raspi': '#A5A5A5'
}

bars1 = ax.bar(x - bar_width, baseline_pc, bar_width,
               label='Baseline PC (.pt)', color=colors['baseline'],
               edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x, jetson_nano, bar_width,
               label='Jetson Nano (engine FP16)', color=colors['jetson'],
               edgecolor='white', linewidth=0.5)
bars3 = ax.bar(x + bar_width, raspberry_pi, bar_width,
               label='Raspberry Pi 4 (tflite FP32)', color=colors['raspi'],
               edgecolor='white', linewidth=0.5)

ax.set_xlabel('Varian Model', fontsize=12, fontweight='bold', labelpad=10)
ax.set_ylabel('mAP50 (%)', fontsize=12, fontweight='bold', labelpad=10)
ax.set_title('Perbandingan mAP50 Antar Platform pada Setiap Varian Model YOLOv8',
             fontsize=13, fontweight='bold', pad=15)

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylim(0, 70)

ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)
ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=7.5, fontweight='bold')

add_value_labels(bars1)
add_value_labels(bars2)
add_value_labels(bars3)

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, 'grafik_map50_perbandingan_platform.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Grafik berhasil disimpan di: {output_path}')
plt.close()
