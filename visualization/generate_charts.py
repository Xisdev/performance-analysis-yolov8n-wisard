"""
Grafik Perbandingan Suhu Maksimal & Daya Rata-rata
Jetson Nano vs Raspberry Pi 4
"""
import matplotlib.pyplot as plt
import numpy as np
import os

# Output directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(ROOT_DIR, "results", "charts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data
models = ['YOLOv8n', 'YOLOv8s', 'YOLOv8m', 'YOLOv8l', 'YOLOv8x']

# Suhu Max
suhu_max_jetson = [41.5, 44.0, 45.0, 45.5, 44.0]
suhu_max_raspi = [45.7, 42.8, 45.2, 47.7, 47.2]

# Daya Avg
daya_avg_jetson = [5951, 6808, 7312, 7637, 7610]
daya_avg_raspi = [3135, 3547, 3197, 3248, 3278]

x = np.arange(len(models))
width = 0.35

# Chart 1: Suhu Max
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, suhu_max_jetson, width, label='Jetson Nano', color='#1f77b4')
rects2 = ax.bar(x + width/2, suhu_max_raspi, width, label='Raspberry Pi 4', color='#ff7f0e')

ax.set_ylabel('Suhu (°C)')
ax.set_title('Grafik Perbandingan Suhu Maksimal (°C) Selama Inferensi')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 60)
ax.legend()

def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1, ax)
autolabel(rects2, ax)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'grafik_suhu_max.png'), dpi=300)
plt.close()

# Chart 2: Daya Avg
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, daya_avg_jetson, width, label='Jetson Nano', color='#2ca02c')
rects2 = ax.bar(x + width/2, daya_avg_raspi, width, label='Raspberry Pi 4', color='#d62728')

ax.set_ylabel('Konsumsi Daya (mW)')
ax.set_title('Grafik Perbandingan Rata-rata Konsumsi Daya (mW)')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 10000)
ax.legend()

autolabel(rects1, ax)
autolabel(rects2, ax)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'grafik_daya_avg.png'), dpi=300)
plt.close()

print(f"Charts saved to: {OUTPUT_DIR}")
