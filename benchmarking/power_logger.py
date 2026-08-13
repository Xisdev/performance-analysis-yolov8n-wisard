"""
==============================================================================
  Power Logger — NVIDIA Jetson Nano (via jtop / jetson-stats)
==============================================================================
  *** SCRIPT INI DIJALANKAN DI NVIDIA JETSON NANO ***

  Mencatat konsumsi daya (power) secara otomatis setiap detik ke file CSV
  menggunakan library jtop (jetson-stats).

  Dijalankan di terminal TERPISAH bersamaan dengan benchmark script.
  Script ini berjalan terus sampai dihentikan dengan Ctrl+C.

  Cara pakai:
      # Jalankan di terminal terpisah SEBELUM benchmark dimulai
      python3 power_logger.py --label yolov8n

      # Atau dengan interval custom (default 1 detik)
      python3 power_logger.py --label yolov8n --interval 0.5

      # Output: power_log_yolov8n_jetson_nano.csv

  Dependensi:
      pip3 install jetson-stats

  CATATAN:
      - jetson-stats sudah include jtop (monitoring tool)
      - Setelah install, RESTART Jetson atau jalankan:
            sudo systemctl restart jtop.service
      - User harus masuk group 'jtop':
            sudo usermod -aG jtop $USER
            (lalu logout & login ulang)
==============================================================================
"""

import os
import sys
import csv
import time
import signal
import argparse
from datetime import datetime
from pathlib import Path

try:
    from jtop import jtop, JtopException
except ImportError:
    print("=" * 60)
    print("[ERROR] jetson-stats belum terinstall!")
    print("        Jalankan:")
    print("          sudo pip3 install jetson-stats")
    print("        Lalu restart service:")
    print("          sudo systemctl restart jtop.service")
    print("=" * 60)
    sys.exit(1)


# ============================================================================
#  KONFIGURASI
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE_NAME = "jetson_nano"
DEFAULT_INTERVAL = 1.0  # detik


# ============================================================================
#  POWER LOGGER
# ============================================================================

class PowerLogger:
    """
    Logger daya menggunakan jtop (jetson-stats).
    
    Mencatat data per detik:
      - timestamp         : waktu pencatatan (ISO format)
      - elapsed_s         : detik sejak mulai
      - power_total_mw    : total daya (milliwatt)
      - power_gpu_mw      : daya GPU (milliwatt)  
      - power_cpu_mw      : daya CPU (milliwatt)
      - power_soc_mw      : daya SoC (milliwatt), jika tersedia
      - gpu_util_pct      : utilisasi GPU (%)
      - cpu_util_pct      : utilisasi CPU rata-rata (%)
      - temp_cpu_c        : suhu CPU (°C)
      - temp_gpu_c        : suhu GPU (°C)
      - ram_used_mb       : RAM terpakai (MB)
      - ram_total_mb      : RAM total (MB)
      - fan_speed_pct     : kecepatan fan (%), jika ada
    """

    def __init__(self, output_path, interval=DEFAULT_INTERVAL):
        self.output_path = output_path
        self.interval = interval
        self._running = False
        self._start_time = None
        self._sample_count = 0

        # CSV header
        self.csv_header = [
            'timestamp', 'elapsed_s',
            'power_total_mw', 'power_gpu_mw', 'power_cpu_mw', 'power_soc_mw',
            'gpu_util_pct', 'cpu_util_pct',
            'temp_cpu_c', 'temp_gpu_c',
            'ram_used_mb', 'ram_total_mb',
            'fan_speed_pct'
        ]

    def _get_power_data(self, jetson):
        """
        Ekstrak data daya dari jtop.
        
        jtop.power mengembalikan dict dengan struktur:
          - 'rail': dict channel daya (GPU, CPU, SOC, dll)
          - 'tot': dict total power {'power': mW, 'avg': mW, ...}
        
        Nama channel bervariasi antar Jetson model:
          Jetson Nano:  POM_5V_GPU, POM_5V_CPU, POM_5V_IN
          Jetson Xavier: GPU, CPU, SOC, CV, DDR, SYS5V
          Jetson Orin:  VDD_GPU_SOC, VDD_CPU_CV, ...
        """
        power = jetson.power

        # Total power
        total_mw = 0
        if 'tot' in power:
            tot = power['tot']
            if isinstance(tot, dict):
                total_mw = tot.get('power', tot.get('avg', 0))
            else:
                total_mw = tot

        # Per-rail power
        gpu_mw = 0
        cpu_mw = 0
        soc_mw = 0

        if 'rail' in power:
            rails = power['rail']
            for name, data in rails.items():
                # Ambil nilai power (bisa dict atau langsung angka)
                if isinstance(data, dict):
                    pw = data.get('power', data.get('avg', 0))
                else:
                    pw = data

                name_upper = name.upper()
                if 'GPU' in name_upper:
                    gpu_mw = pw
                elif 'CPU' in name_upper:
                    cpu_mw = pw
                elif 'SOC' in name_upper or 'SYS' in name_upper:
                    soc_mw = pw

        return total_mw, gpu_mw, cpu_mw, soc_mw

    def _get_gpu_util(self, jetson):
        """Ambil GPU utilization (%)."""
        try:
            gpu = jetson.gpu
            if isinstance(gpu, dict):
                # jetson-stats >= 4.x
                for name, val in gpu.items():
                    if isinstance(val, dict):
                        return val.get('status', {}).get('load', 0)
                    else:
                        return val
            elif isinstance(gpu, (int, float)):
                return gpu
        except Exception:
            pass
        return 0

    def _get_cpu_util(self, jetson):
        """Ambil rata-rata CPU utilization (%)."""
        try:
            cpu = jetson.cpu
            if isinstance(cpu, dict):
                vals = []
                for name, data in cpu.items():
                    if isinstance(data, dict):
                        vals.append(data.get('user', 0) + data.get('system', 0))
                    elif isinstance(data, (int, float)):
                        vals.append(data)
                return sum(vals) / len(vals) if vals else 0
            elif isinstance(cpu, list):
                return sum(cpu) / len(cpu) if cpu else 0
        except Exception:
            pass
        return 0

    def _get_temperatures(self, jetson):
        """Ambil suhu CPU dan GPU (°C)."""
        temp_cpu = 0.0
        temp_gpu = 0.0
        try:
            temp = jetson.temperature
            if isinstance(temp, dict):
                for name, val in temp.items():
                    t = val if isinstance(val, (int, float)) else val.get('temp', 0)
                    name_upper = name.upper()
                    if 'CPU' in name_upper or 'BCPU' in name_upper:
                        temp_cpu = t
                    elif 'GPU' in name_upper:
                        temp_gpu = t
                # Fallback: jika tidak ketemu label spesifik
                if temp_cpu == 0 and temp_gpu == 0:
                    temps = [v if isinstance(v, (int, float)) else v.get('temp', 0)
                             for v in temp.values()]
                    if temps:
                        temp_cpu = max(temps)
                        temp_gpu = max(temps)
        except Exception:
            pass
        return temp_cpu, temp_gpu

    def _get_ram(self, jetson):
        """Ambil RAM used/total (MB)."""
        try:
            ram = jetson.ram
            if isinstance(ram, dict):
                used = ram.get('used', 0)
                total = ram.get('tot', ram.get('total', 0))
                # Bisa dalam bytes atau MB tergantung versi
                if total > 100000:  # kemungkinan dalam bytes/kB
                    used = used / (1024 * 1024)
                    total = total / (1024 * 1024)
                return used, total
        except Exception:
            pass
        return 0, 0

    def _get_fan(self, jetson):
        """Ambil fan speed (%), return 0 jika tidak ada fan."""
        try:
            fan = jetson.fan
            if isinstance(fan, dict):
                for name, data in fan.items():
                    if isinstance(data, dict):
                        return data.get('speed', data.get('auto', 0))
                    else:
                        return data
            elif isinstance(fan, (int, float)):
                return fan
        except Exception:
            pass
        return 0

    def run(self):
        """Mulai logging (blocking). Hentikan dengan Ctrl+C."""
        print(f"\n{'=' * 60}")
        print(f"  POWER LOGGER — {DEVICE_NAME}")
        print(f"{'=' * 60}")
        print(f"  Output     : {self.output_path}")
        print(f"  Interval   : {self.interval}s")
        print(f"  Mulai      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Stop       : Tekan Ctrl+C")
        print(f"{'=' * 60}\n")

        # Tulis header CSV
        with open(self.output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(self.csv_header)

        self._start_time = time.time()
        self._running = True

        try:
            with jtop(interval=self.interval) as jetson:
                print(f"  [OK] jtop terhubung. Logging dimulai...\n")
                print(f"  {'Time':>8s}  {'Total(mW)':>10s}  {'GPU(mW)':>8s}  "
                      f"{'CPU(mW)':>8s}  {'GPU%':>5s}  {'CPU%':>5s}  "
                      f"{'Temp(C)':>7s}  {'RAM(MB)':>8s}")
                print(f"  {'─' * 8}  {'─' * 10}  {'─' * 8}  "
                      f"{'─' * 8}  {'─' * 5}  {'─' * 5}  "
                      f"{'─' * 7}  {'─' * 8}")

                while jetson.ok() and self._running:
                    elapsed = time.time() - self._start_time
                    now = datetime.now().isoformat()

                    # Kumpulkan data
                    total_mw, gpu_mw, cpu_mw, soc_mw = self._get_power_data(jetson)
                    gpu_util = self._get_gpu_util(jetson)
                    cpu_util = self._get_cpu_util(jetson)
                    temp_cpu, temp_gpu = self._get_temperatures(jetson)
                    ram_used, ram_total = self._get_ram(jetson)
                    fan = self._get_fan(jetson)

                    # Tulis ke CSV
                    with open(self.output_path, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            now,
                            f"{elapsed:.1f}",
                            f"{total_mw:.0f}",
                            f"{gpu_mw:.0f}",
                            f"{cpu_mw:.0f}",
                            f"{soc_mw:.0f}",
                            f"{gpu_util:.1f}",
                            f"{cpu_util:.1f}",
                            f"{temp_cpu:.1f}",
                            f"{temp_gpu:.1f}",
                            f"{ram_used:.0f}",
                            f"{ram_total:.0f}",
                            f"{fan:.0f}"
                        ])

                    self._sample_count += 1

                    # Print ke terminal (setiap record)
                    print(f"  {elapsed:7.1f}s  {total_mw:10.0f}  {gpu_mw:8.0f}  "
                          f"{cpu_mw:8.0f}  {gpu_util:5.1f}  {cpu_util:5.1f}  "
                          f"{max(temp_cpu, temp_gpu):7.1f}  "
                          f"{ram_used:4.0f}/{ram_total:.0f}")

        except JtopException as e:
            print(f"\n[ERROR] jtop error: {e}")
            print("        Coba:")
            print("          sudo systemctl restart jtop.service")
            sys.exit(1)
        except KeyboardInterrupt:
            pass

        # Ringkasan saat selesai
        duration = time.time() - self._start_time
        print(f"\n{'=' * 60}")
        print(f"  LOGGING SELESAI")
        print(f"{'=' * 60}")
        print(f"  Durasi     : {duration:.1f}s ({duration/60:.1f} menit)")
        print(f"  Samples    : {self._sample_count}")
        print(f"  Output     : {self.output_path}")
        print(f"{'=' * 60}")


# ============================================================================
#  MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Power Logger untuk NVIDIA Jetson Nano (via jtop)"
    )
    parser.add_argument(
        "--label", "-l", type=str, required=True,
        help="Label untuk file output, contoh: 'yolov8n' → power_log_yolov8n_jetson_nano.csv"
    )
    parser.add_argument(
        "--interval", "-i", type=float, default=DEFAULT_INTERVAL,
        help=f"Interval sampling dalam detik (default: {DEFAULT_INTERVAL})"
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default=SCRIPT_DIR,
        help=f"Folder output CSV (default: {SCRIPT_DIR})"
    )

    args = parser.parse_args()

    # Buat nama file output
    output_path = os.path.join(
        args.output_dir,
        f"power_log_{args.label}_{DEVICE_NAME}.csv"
    )

    # Handle Ctrl+C dengan graceful
    logger = PowerLogger(output_path, interval=args.interval)

    def signal_handler(sig, frame):
        logger._running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.run()


if __name__ == "__main__":
    main()
