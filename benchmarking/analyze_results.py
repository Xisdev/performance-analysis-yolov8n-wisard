"""
==============================================================================
  Analisis Hasil Benchmark YOLOv8 — Jetson Nano vs Raspberry Pi 4
==============================================================================
  Script ini dijalankan di PC setelah mengumpulkan CSV dari kedua device.

  Cara pakai:
      python analyze_results.py
      python analyze_results.py --jetson-dir ./jetson_nano/ --raspi-dir ./raspi4/
      python analyze_results.py --output ./results/

  Input yang diharapkan:
      jetson_nano/
      ├── benchmark_fps_latency_{model}_jetson_nano.csv
      ├── benchmark_resource_{model}_jetson_nano.csv
      └── benchmark_summary_{model}_jetson_nano.json

      raspi4/
      ├── benchmark_fps_latency_{model}_raspi4.csv
      ├── benchmark_resource_{model}_raspi4.csv
      └── benchmark_summary_{model}_raspi4.json

  Output:
      results/
      ├── tabel_kinerja_realtime.csv        (Tabel 4.2)
      ├── tabel_profiling_resource.csv      (Tabel 4.4)
      ├── chart_fps_comparison.png
      ├── chart_latency_breakdown.png
      ├── chart_latency_over_time.png
      ├── chart_fps_vs_model_size.png
      └── full_report.json
==============================================================================
"""

import os
import sys
import csv
import json
import glob
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    print("[WARN] matplotlib tidak tersedia. Grafik tidak akan dibuat.")
    print("       Install: pip install matplotlib")
    HAS_MATPLOTLIB = False


# ============================================================================
#  KONFIGURASI
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_JETSON_DIR = os.path.join(SCRIPT_DIR, "jetson_nano")
DEFAULT_RASPI_DIR = os.path.join(SCRIPT_DIR, "raspberry_pi_4")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results")

# Urutan model yang diinginkan
MODEL_ORDER = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']

# Warna untuk grafik
COLORS = {
    'jetson_nano': '#76b900',  # NVIDIA Green
    'raspi4': '#c51a4a',       # Raspberry Pi Red
    'preprocess': '#58a6ff',
    'inference': '#f85149',
    'postprocess': '#3fb950',
    'total': '#d29922',
}


# ============================================================================
#  PLOT STYLE
# ============================================================================

def setup_plot_style():
    """Configure premium dark matplotlib style."""
    if not HAS_MATPLOTLIB:
        return
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


# ============================================================================
#  LOAD DATA
# ============================================================================

def load_summary_files(directory, device_name):
    """Load semua file benchmark_summary_*.json dari suatu folder."""
    results = {}
    pattern = os.path.join(directory, f"benchmark_summary_*_{device_name}.json")
    files = sorted(glob.glob(pattern))

    for filepath in files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            model_name = data.get('model_name', Path(filepath).stem)
            results[model_name] = data
            print(f"  ✓ Loaded: {os.path.basename(filepath)}")
        except Exception as e:
            print(f"  ✗ Error loading {filepath}: {e}")

    return results


def load_fps_csv(directory, device_name):
    """Load semua file benchmark_fps_latency CSV."""
    results = {}
    pattern = os.path.join(directory, f"benchmark_fps_latency_*_{device_name}.csv")
    files = sorted(glob.glob(pattern))

    for filepath in files:
        try:
            rows = []
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('is_warmup', 'false') == 'false':
                        try:
                            rows.append({
                                'frame_idx': int(row['frame_idx']),
                                'preprocess_ms': float(row['preprocess_ms']),
                                'inference_ms': float(row['inference_ms']),
                                'postprocess_ms': float(row['postprocess_ms']),
                                'total_ms': float(row['total_ms']),
                                'fps': float(row['fps']),
                                'num_detections': int(row['num_detections']),
                            })
                        except (ValueError, KeyError):
                            continue  # Skip error rows

            # Extract model name from filename
            basename = Path(filepath).stem
            model_name = basename.replace(f"benchmark_fps_latency_", "").replace(f"_{device_name}", "")
            results[model_name] = rows
            print(f"  ✓ Loaded CSV: {os.path.basename(filepath)} ({len(rows)} frames)")
        except Exception as e:
            print(f"  ✗ Error loading {filepath}: {e}")

    return results


def load_resource_csv(directory, device_name):
    """Load semua file benchmark_resource CSV."""
    results = {}
    pattern = os.path.join(directory, f"benchmark_resource_*_{device_name}.csv")
    files = sorted(glob.glob(pattern))

    for filepath in files:
        try:
            rows = []
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        rows.append({
                            'elapsed_s': float(row['elapsed_s']),
                            'cpu_percent': float(row['cpu_percent']),
                            'gpu_percent': float(row['gpu_percent']),
                            'ram_used_mb': float(row['ram_used_mb']),
                            'ram_total_mb': float(row['ram_total_mb']),
                            'temperature_c': float(row['temperature_c']),
                        })
                    except (ValueError, KeyError):
                        continue

            basename = Path(filepath).stem
            model_name = basename.replace(f"benchmark_resource_", "").replace(f"_{device_name}", "")
            results[model_name] = rows
            print(f"  ✓ Loaded Resource CSV: {os.path.basename(filepath)} ({len(rows)} samples)")
        except Exception as e:
            print(f"  ✗ Error loading {filepath}: {e}")

    return results


def sort_model_key(model_name):
    """Key function untuk mengurutkan model sesuai urutan standar."""
    name_lower = model_name.lower()
    for idx, variant in enumerate(MODEL_ORDER):
        if variant in name_lower:
            return idx
    return 99


# ============================================================================
#  GENERATE TABLES
# ============================================================================

def generate_kinerja_table(jetson_summaries, raspi_summaries, output_dir):
    """
    Generate Tabel 4.2: Kinerja Real-Time (FPS & Latency).
    """
    csv_path = os.path.join(output_dir, "tabel_kinerja_realtime.csv")

    # Kumpulkan semua model
    all_models = sorted(
        set(list(jetson_summaries.keys()) + list(raspi_summaries.keys())),
        key=sort_model_key
    )

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Model', 'Device', 'Status',
            'Model Size (MB)',
            'Preprocess (ms)', 'Inference (ms)', 'Postprocess (ms)', 'Total Latency (ms)',
            'FPS Mean', 'FPS Median', 'FPS Min', 'FPS Max',
        ])

        for model in all_models:
            for device, summaries in [('Jetson Nano', jetson_summaries), ('Raspberry Pi 4', raspi_summaries)]:
                if model not in summaries:
                    continue

                s = summaries[model]
                status = s.get('status', 'OK')

                if status == 'OK' and 'latency' in s:
                    writer.writerow([
                        model, device, status,
                        s.get('model_size_mb', ''),
                        s['latency']['preprocess_ms']['mean'],
                        s['latency']['inference_ms']['mean'],
                        s['latency']['postprocess_ms']['mean'],
                        s['latency']['total_ms']['mean'],
                        s['fps']['mean'],
                        s['fps']['median'],
                        s['fps']['min'],
                        s['fps']['max'],
                    ])
                else:
                    error = s.get('error', status)
                    writer.writerow([
                        model, device, status,
                        s.get('model_size_mb', ''),
                        error, '', '', '',
                        '', '', '', '',
                    ])

    print(f"  📄 {csv_path}")
    return csv_path


def generate_resource_table(jetson_resources, raspi_resources, output_dir):
    """
    Generate Tabel 4.4: Profiling Sumber Daya.
    """
    csv_path = os.path.join(output_dir, "tabel_profiling_resource.csv")

    all_models = sorted(
        set(list(jetson_resources.keys()) + list(raspi_resources.keys())),
        key=sort_model_key
    )

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Model', 'Device',
            'CPU Avg (%)', 'CPU Max (%)',
            'GPU Avg (%)', 'GPU Max (%)',
            'RAM Avg (MB)', 'RAM Max (MB)',
            'Temp Avg (°C)', 'Temp Max (°C)',
        ])

        for model in all_models:
            for device, resources in [('Jetson Nano', jetson_resources), ('Raspberry Pi 4', raspi_resources)]:
                if model not in resources or not resources[model]:
                    continue

                rows = resources[model]
                cpus = [r['cpu_percent'] for r in rows]
                gpus = [r['gpu_percent'] for r in rows]
                rams = [r['ram_used_mb'] for r in rows]
                temps = [r['temperature_c'] for r in rows]

                writer.writerow([
                    model, device,
                    f"{np.mean(cpus):.1f}", f"{np.max(cpus):.1f}",
                    f"{np.mean(gpus):.1f}", f"{np.max(gpus):.1f}",
                    f"{np.mean(rams):.0f}", f"{np.max(rams):.0f}",
                    f"{np.mean(temps):.1f}", f"{np.max(temps):.1f}",
                ])

    print(f"  📄 {csv_path}")
    return csv_path


# ============================================================================
#  GENERATE CHARTS
# ============================================================================

def chart_fps_comparison(jetson_summaries, raspi_summaries, output_dir):
    """Bar chart: FPS comparison across models and devices."""
    if not HAS_MATPLOTLIB:
        return

    models = sorted(
        set(list(jetson_summaries.keys()) + list(raspi_summaries.keys())),
        key=sort_model_key
    )

    # Filter hanya model yang punya data FPS
    models_ok = []
    for m in models:
        j = jetson_summaries.get(m, {})
        r = raspi_summaries.get(m, {})
        if (j.get('status') == 'OK') or (r.get('status') == 'OK'):
            models_ok.append(m)

    if not models_ok:
        print("  [SKIP] Tidak ada data FPS untuk chart")
        return

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(models_ok))
    width = 0.35

    jetson_fps = []
    raspi_fps = []
    for m in models_ok:
        j = jetson_summaries.get(m, {})
        r = raspi_summaries.get(m, {})
        jetson_fps.append(j.get('fps', {}).get('mean', 0) if j.get('status') == 'OK' else 0)
        raspi_fps.append(r.get('fps', {}).get('mean', 0) if r.get('status') == 'OK' else 0)

    bars1 = ax.bar(x - width/2, jetson_fps, width,
                   label='Jetson Nano', color=COLORS['jetson_nano'],
                   edgecolor='none', alpha=0.9)
    bars2 = ax.bar(x + width/2, raspi_fps, width,
                   label='Raspberry Pi 4', color=COLORS['raspi4'],
                   edgecolor='none', alpha=0.9)

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                        f'{h:.1f}', ha='center', va='bottom',
                        fontsize=10, fontweight='bold', color='white')

    # Label OOM models
    for i, m in enumerate(models_ok):
        r = raspi_summaries.get(m, {})
        if r.get('status', '').startswith('OOM'):
            ax.text(x[i] + width/2, 0.5, 'OOM', ha='center', va='bottom',
                    fontsize=9, color=COLORS['raspi4'], fontweight='bold',
                    fontstyle='italic')

    ax.set_xlabel('Model', fontsize=13)
    ax.set_ylabel('FPS (frames/second)', fontsize=13)
    ax.set_title('Perbandingan FPS — Jetson Nano vs Raspberry Pi 4',
                 fontsize=16, fontweight='bold', color='white')
    ax.set_xticks(x)

    # Buat label model lebih pendek
    short_labels = [m.replace('best_', '').replace('_wisard_ir', '').replace('_fp16', '').replace('_fp32', '')
                    for m in models_ok]
    ax.set_xticklabels(short_labels, fontsize=11)
    ax.legend(facecolor='#161b22', edgecolor='#30363d', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    path = os.path.join(output_dir, 'chart_fps_comparison.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  📊 {path}")


def chart_latency_breakdown(jetson_summaries, raspi_summaries, output_dir):
    """Stacked bar chart: Latency breakdown per model per device."""
    if not HAS_MATPLOTLIB:
        return

    models = sorted(
        set(list(jetson_summaries.keys()) + list(raspi_summaries.keys())),
        key=sort_model_key
    )

    # Filter hanya yang OK
    models_ok = [m for m in models
                 if jetson_summaries.get(m, {}).get('status') == 'OK'
                 or raspi_summaries.get(m, {}).get('status') == 'OK']

    if not models_ok:
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, (device_label, summaries, device_key) in zip(axes, [
        ('Jetson Nano', jetson_summaries, 'jetson_nano'),
        ('Raspberry Pi 4', raspi_summaries, 'raspi4'),
    ]):
        ok_models = [m for m in models_ok if summaries.get(m, {}).get('status') == 'OK']
        if not ok_models:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=16, color='#8b949e')
            ax.set_title(device_label, fontweight='bold', color='white')
            continue

        pre = [summaries[m]['latency']['preprocess_ms']['mean'] for m in ok_models]
        inf = [summaries[m]['latency']['inference_ms']['mean'] for m in ok_models]
        post = [summaries[m]['latency']['postprocess_ms']['mean'] for m in ok_models]

        x = np.arange(len(ok_models))

        ax.bar(x, pre, 0.5, label='Preprocess', color=COLORS['preprocess'], alpha=0.9)
        ax.bar(x, inf, 0.5, bottom=pre, label='Inference', color=COLORS['inference'], alpha=0.9)
        ax.bar(x, post, 0.5, bottom=[p + i for p, i in zip(pre, inf)],
               label='Postprocess', color=COLORS['postprocess'], alpha=0.9)

        # Total label
        totals = [p + i + pp for p, i, pp in zip(pre, inf, post)]
        for xi, t in zip(x, totals):
            ax.text(xi, t + 1, f'{t:.1f}ms', ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color='white')

        short_labels = [m.replace('best_', '').replace('_wisard_ir', '').replace('_fp16', '').replace('_fp32', '')
                        for m in ok_models]
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, fontsize=10)
        ax.set_ylabel('Latency (ms)', fontsize=12)
        ax.set_title(device_label, fontweight='bold', color='white', fontsize=14)
        ax.legend(facecolor='#161b22', edgecolor='#30363d', fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Breakdown Latency per Tahap — Jetson Nano vs Raspberry Pi 4',
                 fontsize=16, fontweight='bold', color='white', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'chart_latency_breakdown.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  📊 {path}")


def chart_latency_over_time(jetson_csvs, raspi_csvs, output_dir):
    """Line chart: Latency over time (stability analysis)."""
    if not HAS_MATPLOTLIB:
        return

    all_models = sorted(
        set(list(jetson_csvs.keys()) + list(raspi_csvs.keys())),
        key=sort_model_key
    )

    if not all_models:
        return

    # Pilih model pertama yang punya data untuk kedua device (atau satu saja)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=False)

    for ax, (device_label, csvs) in zip(axes, [
        ('Jetson Nano', jetson_csvs),
        ('Raspberry Pi 4', raspi_csvs),
    ]):
        has_data = False
        for model in all_models:
            if model in csvs and csvs[model]:
                data = csvs[model]
                latencies = [r['total_ms'] for r in data]
                if len(latencies) > 20:
                    # Smoothing
                    window = min(20, len(latencies) // 5)
                    smoothed = np.convolve(latencies, np.ones(window)/window, mode='valid')
                    ax.plot(range(len(latencies)), latencies, alpha=0.15, linewidth=0.5)
                    ax.plot(range(window-1, len(latencies)), smoothed, linewidth=2,
                            label=model.replace('best_', '').replace('_wisard_ir', ''))
                else:
                    ax.plot(latencies, linewidth=1.5,
                            label=model.replace('best_', '').replace('_wisard_ir', ''))
                has_data = True

        if not has_data:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=16, color='#8b949e')

        ax.set_xlabel('Frame Index', fontsize=12)
        ax.set_ylabel('Total Latency (ms)', fontsize=12)
        ax.set_title(device_label, fontweight='bold', color='white')
        ax.legend(facecolor='#161b22', edgecolor='#30363d', fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Stabilitas Latency — Jetson Nano vs Raspberry Pi 4',
                 fontsize=16, fontweight='bold', color='white', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'chart_latency_over_time.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  📊 {path}")


def chart_fps_vs_model_size(jetson_summaries, raspi_summaries, output_dir):
    """Scatter plot: FPS vs Model Size (trade-off analysis)."""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    # Jetson Nano
    for model, s in jetson_summaries.items():
        if s.get('status') != 'OK':
            continue
        size = s.get('model_size_mb', 0)
        fps = s['fps']['mean']
        label = model.replace('best_', '').replace('_wisard_ir', '').replace('_fp16', '')
        ax.scatter(size, fps, s=120, color=COLORS['jetson_nano'], edgecolors='white',
                   linewidth=1.5, zorder=5, alpha=0.9)
        ax.annotate(label, (size, fps), textcoords="offset points",
                    xytext=(8, 5), fontsize=9, color=COLORS['jetson_nano'])

    # Raspberry Pi 4
    for model, s in raspi_summaries.items():
        if s.get('status') != 'OK':
            continue
        size = s.get('model_size_mb', 0)
        fps = s['fps']['mean']
        label = model.replace('best_', '').replace('_wisard_ir', '').replace('_fp32', '')
        ax.scatter(size, fps, s=120, color=COLORS['raspi4'], edgecolors='white',
                   linewidth=1.5, zorder=5, alpha=0.9, marker='s')
        ax.annotate(label, (size, fps), textcoords="offset points",
                    xytext=(8, -10), fontsize=9, color=COLORS['raspi4'])

    # Legend manual
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['jetson_nano'],
               markersize=10, label='Jetson Nano (TensorRT)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['raspi4'],
               markersize=10, label='Raspberry Pi 4 (TFLite)'),
    ]
    ax.legend(handles=legend_elements, facecolor='#161b22', edgecolor='#30363d', fontsize=11)

    ax.set_xlabel('Model Size (MB)', fontsize=13)
    ax.set_ylabel('FPS (frames/second)', fontsize=13)
    ax.set_title('Trade-off: FPS vs Model Size',
                 fontsize=16, fontweight='bold', color='white')
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, 'chart_fps_vs_model_size.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  📊 {path}")


# ============================================================================
#  MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analisis hasil benchmark YOLOv8 — Jetson Nano vs Raspberry Pi 4"
    )
    parser.add_argument(
        "--jetson-dir", type=str, default=DEFAULT_JETSON_DIR,
        help=f"Folder hasil benchmark Jetson Nano (default: {DEFAULT_JETSON_DIR})"
    )
    parser.add_argument(
        "--raspi-dir", type=str, default=DEFAULT_RASPI_DIR,
        help=f"Folder hasil benchmark Raspberry Pi 4 (default: {DEFAULT_RASPI_DIR})"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Folder output analisis (default: {DEFAULT_OUTPUT_DIR})"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("  ANALISIS HASIL BENCHMARK")
    print("  YOLOv8 — Jetson Nano vs Raspberry Pi 4")
    print("=" * 70)

    if HAS_MATPLOTLIB:
        setup_plot_style()

    os.makedirs(args.output, exist_ok=True)

    # ── Load Data ──────────────────────────────────────────────────────
    print(f"\n[1/4] Loading data...")

    print(f"\n  Jetson Nano ({args.jetson_dir}):")
    jetson_summaries = load_summary_files(args.jetson_dir, "jetson_nano")
    jetson_csvs = load_fps_csv(args.jetson_dir, "jetson_nano")
    jetson_resources = load_resource_csv(args.jetson_dir, "jetson_nano")

    print(f"\n  Raspberry Pi 4 ({args.raspi_dir}):")
    raspi_summaries = load_summary_files(args.raspi_dir, "raspi4")
    raspi_csvs = load_fps_csv(args.raspi_dir, "raspi4")
    raspi_resources = load_resource_csv(args.raspi_dir, "raspi4")

    total_loaded = len(jetson_summaries) + len(raspi_summaries)
    if total_loaded == 0:
        print("\n[ERROR] Tidak ada data benchmark yang ditemukan!")
        print(f"        Pastikan file CSV/JSON ada di:")
        print(f"        - {args.jetson_dir}")
        print(f"        - {args.raspi_dir}")
        print(f"\n        Format file yang diharapkan:")
        print(f"        - benchmark_summary_*_jetson_nano.json")
        print(f"        - benchmark_summary_*_raspi4.json")
        sys.exit(1)

    print(f"\n  Total: {len(jetson_summaries)} model Jetson + {len(raspi_summaries)} model RPi4")

    # ── Generate Tables ────────────────────────────────────────────────
    print(f"\n[2/4] Generating tables...")
    generate_kinerja_table(jetson_summaries, raspi_summaries, args.output)
    generate_resource_table(jetson_resources, raspi_resources, args.output)

    # ── Generate Charts ────────────────────────────────────────────────
    print(f"\n[3/4] Generating charts...")
    chart_fps_comparison(jetson_summaries, raspi_summaries, args.output)
    chart_latency_breakdown(jetson_summaries, raspi_summaries, args.output)
    chart_latency_over_time(jetson_csvs, raspi_csvs, args.output)
    chart_fps_vs_model_size(jetson_summaries, raspi_summaries, args.output)

    # ── Full Report ────────────────────────────────────────────────────
    print(f"\n[4/4] Generating full report...")

    report = {
        'title': 'YOLOv8 Benchmark Report — Jetson Nano vs Raspberry Pi 4',
        'generated_at': str(np.datetime64('now')),
        'jetson_nano': {
            'models': len(jetson_summaries),
            'results': jetson_summaries,
        },
        'raspi4': {
            'models': len(raspi_summaries),
            'results': raspi_summaries,
        },
    }

    report_path = os.path.join(args.output, 'full_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"  📄 {report_path}")

    # ── Print Quick Summary ────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  RINGKASAN CEPAT")
    print(f"{'=' * 70}")

    print(f"\n  ── Jetson Nano ──")
    for model, s in sorted(jetson_summaries.items(), key=lambda x: sort_model_key(x[0])):
        if s.get('status') == 'OK':
            print(f"  {model:<35s} FPS: {s['fps']['mean']:>7.2f}  Latency: {s['latency']['total_ms']['mean']:>8.2f}ms")
        else:
            print(f"  {model:<35s} {s.get('status', 'ERROR')}")

    print(f"\n  ── Raspberry Pi 4 ──")
    for model, s in sorted(raspi_summaries.items(), key=lambda x: sort_model_key(x[0])):
        if s.get('status') == 'OK':
            print(f"  {model:<35s} FPS: {s['fps']['mean']:>7.2f}  Latency: {s['latency']['total_ms']['mean']:>8.2f}ms")
        else:
            print(f"  {model:<35s} {s.get('status', 'ERROR')}")

    print(f"\n{'=' * 70}")
    print(f"  Output disimpan di: {args.output}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
