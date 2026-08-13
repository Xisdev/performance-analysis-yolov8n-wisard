"""
Ekstrak data benchmark RPi4: Latency, FPS, Resource (Average & Max).
Warmup frames diexclude dari perhitungan.
"""
import os
import csv
import json

DATA_DIR = r"D:\RISET\drone-wisard\test_for_device\raspberry_pi_4\data_hasil_raspi"
OUTPUT_DIR = r"D:\RISET\drone-wisard\test_for_device\raspberry_pi_4\data_hasil_raspi\hasil"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']

def read_csv(path):
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

# ============================================================
# 1. LATENCY & FPS
# ============================================================
latency_results = {}

for model in MODELS:
    csv_file = os.path.join(DATA_DIR, f"benchmark_fps_latency_best_{model}_fp32_raspi4.csv")
    if not os.path.exists(csv_file):
        print(f"[SKIP] {csv_file} not found")
        continue

    rows = read_csv(csv_file)
    # Filter: exclude warmup AND exclude error rows
    data = [r for r in rows if r.get('is_warmup', 'true') == 'false'
            and r.get('preprocess_ms', 'ERROR') not in ('ERROR', 'OOM')]

    if not data:
        print(f"[WARN] No valid data for {model}")
        continue

    preprocess = [float(r['preprocess_ms']) for r in data]
    inference = [float(r['inference_ms']) for r in data]
    postprocess = [float(r['postprocess_ms']) for r in data]
    total = [float(r['total_ms']) for r in data]
    fps = [float(r['fps']) for r in data]

    latency_results[model] = {
        'frames': len(data),
        'preprocess_ms': round(sum(preprocess) / len(preprocess), 2),
        'inference_ms': round(sum(inference) / len(inference), 2),
        'postprocess_ms': round(sum(postprocess) / len(postprocess), 2),
        'total_ms': round(sum(total) / len(total), 2),
        'fps': round(sum(fps) / len(fps), 2),
    }

# ============================================================
# 2. RESOURCE (Average & Max)
# ============================================================
resource_avg = {}
resource_max = {}

for model in MODELS:
    csv_file = os.path.join(DATA_DIR, f"benchmark_resource_best_{model}_fp32_raspi4.csv")
    if not os.path.exists(csv_file):
        print(f"[SKIP] {csv_file} not found")
        continue

    rows = read_csv(csv_file)
    if not rows:
        print(f"[WARN] No resource data for {model}")
        continue

    cpu = [float(r['cpu_percent']) for r in rows]
    gpu = [float(r['gpu_percent']) for r in rows]
    ram_used = [float(r['ram_used_mb']) for r in rows]
    ram_total = [float(r['ram_total_mb']) for r in rows]
    temp = [float(r['temperature_c']) for r in rows]

    n = len(rows)
    resource_avg[model] = {
        'samples': n,
        'cpu_percent': round(sum(cpu) / n, 2),
        'gpu_percent': round(sum(gpu) / n, 2),
        'ram_used_mb': round(sum(ram_used) / n, 2),
        'ram_total_mb': round(sum(ram_total) / n, 2),
        'temperature_c': round(sum(temp) / n, 2),
    }
    resource_max[model] = {
        'samples': n,
        'cpu_percent': round(max(cpu), 2),
        'gpu_percent': round(max(gpu), 2),
        'ram_used_mb': round(max(ram_used), 2),
        'ram_total_mb': round(max(ram_total), 2),
        'temperature_c': round(max(temp), 2),
    }

# ============================================================
# SAVE: CSV + JSON
# ============================================================

# --- Latency & FPS CSV ---
latency_csv = os.path.join(OUTPUT_DIR, "average_latency_fps_raspi4.csv")
with open(latency_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['model', 'frames', 'preprocess_ms', 'inference_ms', 'postprocess_ms', 'total_ms', 'fps'])
    for model in MODELS:
        if model in latency_results:
            d = latency_results[model]
            writer.writerow([model, d['frames'], d['preprocess_ms'], d['inference_ms'],
                           d['postprocess_ms'], d['total_ms'], d['fps']])

# --- Resource AVG CSV ---
avg_csv = os.path.join(OUTPUT_DIR, "average_resource_raspi4.csv")
with open(avg_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['model', 'samples', 'cpu_percent', 'gpu_percent', 'ram_used_mb', 'ram_total_mb', 'temperature_c'])
    for model in MODELS:
        if model in resource_avg:
            d = resource_avg[model]
            writer.writerow([model, d['samples'], d['cpu_percent'], d['gpu_percent'],
                           d['ram_used_mb'], d['ram_total_mb'], d['temperature_c']])

# --- Resource MAX CSV ---
max_csv = os.path.join(OUTPUT_DIR, "max_resource_raspi4.csv")
with open(max_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['model', 'samples', 'cpu_percent', 'gpu_percent', 'ram_used_mb', 'ram_total_mb', 'temperature_c'])
    for model in MODELS:
        if model in resource_max:
            d = resource_max[model]
            writer.writerow([model, d['samples'], d['cpu_percent'], d['gpu_percent'],
                           d['ram_used_mb'], d['ram_total_mb'], d['temperature_c']])

# --- Combined JSON ---
combined = {
    'average_latency_fps': latency_results,
    'average_resource': resource_avg,
    'max_resource': resource_max,
}
json_path = os.path.join(OUTPUT_DIR, "benchmark_summary_raspi4.json")
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(combined, f, indent=2, ensure_ascii=False)

# ============================================================
# PRINT TABLES
# ============================================================
print("=" * 85)
print("  AVERAGE BENCHMARK LATENCY & FPS (excl. warmup)")
print("=" * 85)
print(f"  {'Model':<12s} {'Frames':>7s} {'Preproc(ms)':>12s} {'Infer(ms)':>11s} {'Postproc(ms)':>13s} {'Total(ms)':>11s} {'FPS':>8s}")
print(f"  {'-'*12} {'-'*7} {'-'*12} {'-'*11} {'-'*13} {'-'*11} {'-'*8}")
for model in MODELS:
    if model in latency_results:
        d = latency_results[model]
        print(f"  {model:<12s} {d['frames']:>7d} {d['preprocess_ms']:>12.2f} {d['inference_ms']:>11.2f} "
              f"{d['postprocess_ms']:>13.2f} {d['total_ms']:>11.2f} {d['fps']:>8.2f}")

print(f"\n{'='*85}")
print("  AVERAGE BENCHMARK RESOURCE")
print("=" * 85)
print(f"  {'Model':<12s} {'Samples':>8s} {'CPU(%)':>8s} {'GPU(%)':>8s} {'RAM Used(MB)':>13s} {'RAM Total(MB)':>14s} {'Temp(C)':>9s}")
print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*13} {'-'*14} {'-'*9}")
for model in MODELS:
    if model in resource_avg:
        d = resource_avg[model]
        print(f"  {model:<12s} {d['samples']:>8d} {d['cpu_percent']:>8.2f} {d['gpu_percent']:>8.2f} "
              f"{d['ram_used_mb']:>13.2f} {d['ram_total_mb']:>14.2f} {d['temperature_c']:>9.2f}")

print(f"\n{'='*85}")
print("  MAX BENCHMARK RESOURCE")
print("=" * 85)
print(f"  {'Model':<12s} {'Samples':>8s} {'CPU(%)':>8s} {'GPU(%)':>8s} {'RAM Used(MB)':>13s} {'RAM Total(MB)':>14s} {'Temp(C)':>9s}")
print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*13} {'-'*14} {'-'*9}")
for model in MODELS:
    if model in resource_max:
        d = resource_max[model]
        print(f"  {model:<12s} {d['samples']:>8d} {d['cpu_percent']:>8.2f} {d['gpu_percent']:>8.2f} "
              f"{d['ram_used_mb']:>13.2f} {d['ram_total_mb']:>14.2f} {d['temperature_c']:>9.2f}")

print(f"\n{'='*85}")
print(f"  Output saved to: {OUTPUT_DIR}")
print(f"    - average_latency_fps_raspi4.csv")
print(f"    - average_resource_raspi4.csv")
print(f"    - max_resource_raspi4.csv")
print(f"    - benchmark_summary_raspi4.json")
print(f"{'='*85}")
