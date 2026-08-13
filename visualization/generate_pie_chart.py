"""
Grafik Lingkaran (Pie Chart) - Pembagian Dataset WiSARD IR
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

# Output directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(ROOT_DIR, "results", "charts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data
labels = ['Training Set\n31.554 citra (58,6%)',
          'Validation Set\n16.302 citra (30,3%)',
          'Testing Set\n5.950 citra (11,1%)']
sizes = [58.6, 30.3, 11.1]
colors = ['#2563EB', '#F59E0B', '#10B981']
explode = (0.03, 0.03, 0.03)

# Setup figure
fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')

wedges, texts, autotexts = ax.pie(
    sizes,
    explode=explode,
    labels=labels,
    colors=colors,
    autopct='%1.1f%%',
    startangle=90,
    pctdistance=0.55,
    labeldistance=1.18,
    shadow=False,
    wedgeprops=dict(edgecolor='white', linewidth=2.5),
    textprops=dict(fontsize=11, fontweight='medium'),
)

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(13)
    autotext.set_fontweight('bold')

ax.set_title(
    'Pembagian Dataset WiSARD IR\n(Total: 53.806 Citra Thermal Inframerah)',
    fontsize=14, fontweight='bold', pad=20, color='#1F2937'
)

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, 'pie_chart_dataset_split.png')
plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Pie chart saved: {output_path}")
plt.close()
