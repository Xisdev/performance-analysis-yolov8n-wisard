# PANDUAN EKSEKUSI LENGKAP
# Benchmark YOLOv8 (n/s/m/l/x) pada Jetson Nano & Raspberry Pi 4
# ================================================================
# Tanggal: 15 Juni 2026 (Diperbarui: 15 Juli 2026)
# Folder kerja: D:\RISET\drone-wisard\test_for_device
# Tabel data  : D:\RISET\drone-wisard\penulisan\tabel_data.xlsx
# ================================================================
#
# DAFTAR ISI:
#   FASE 1 - Persiapan di PC (Sebelum ke Device)        [DONE]
#   FASE 2 - Transfer File ke Device (via Flashdisk)
#   FASE 3 - Eksekusi di NVIDIA Jetson Nano (layar langsung)
#   FASE 4 - Eksekusi di Raspberry Pi 4 (layar langsung)
#   FASE 5 - Salin Hasil dari Flashdisk ke PC
#   FASE 6 - Analisis & Generate Tabel di PC
#
# METODE TRANSFER: Flashdisk USB (bukan SSH/SCP)
# METODE EKSEKUSI: Terminal/cmd langsung di layar perangkat (bukan SSH)
#
# STRUKTUR FOLDER DEVICE (self-contained, tinggal copy 1 folder):
#
#   jetson_nano/                    <-- Copy seluruh folder ini ke Jetson
#   ├── benchmark_jetson.py         script benchmark FPS & latency
#   ├── evaluate_accuracy.py        script evaluasi akurasi
#   ├── convert_onnx_to_trt.py      script konversi ONNX -> TensorRT
#   ├── onnx_model/                 5 file model .onnx
#   ├── models/                     (kosong, diisi .engine setelah konversi)
#   ├── test/                       5950 gambar test
#   └── test_labels/                5950 label ground truth
#
#   raspberry_pi_4/                 <-- Copy seluruh folder ini ke RPi4
#   ├── benchmark_raspi.py          script benchmark FPS & latency
#   ├── evaluate_accuracy.py        script evaluasi akurasi
#   ├── tflite_model/               5 file model .tflite (sudah siap pakai)
#   ├── test/                       5950 gambar test
#   └── test_labels/                5950 label ground truth
#
# CATATAN PENTING:
#   - SEMUA model (n/s/m/l/x) sudah selesai di-train dan optimizer statenya
#     telah di-strip. Model siap deploy ada di folder: kumpulan_model_ready/
#   - Model .onnx dan .tflite sudah dikonversi dan disimpan di folder masing-masing.
#   - Gambar test dan label SAMA PERSIS untuk kedua device (apple-to-apple).
# ================================================================


################################################################################
#                                                                              #
#                    FASE 1: PERSIAPAN DI PC                                   #
#             (Dijalankan di PC Windows - Command Prompt / PowerShell)         #
#                                                                              #
################################################################################


# ===========================================================================
# [DONE] LANGKAH 1.1 - Gambar Test (SUDAH DISIAPKAN)
# ===========================================================================
# Penjelasan:
#   Gambar test menggunakan SELURUH 5950 gambar dari dataset WiSARD test set,
#   yang sudah disalin ke folder test_for_device.
#
#   Struktur folder:
#     test_for_device/test/         = 5950 gambar test (.jpg)       [OK]
#     test_for_device/test_labels/  = 5950 label ground truth (.txt) [OK]
#
#   Gambar yang SAMA PERSIS ini sudah disalin ke folder:
#     jetson_nano/test/         dan  jetson_nano/test_labels/
#     raspberry_pi_4/test/      dan  raspberry_pi_4/test_labels/
# ---------------------------------------------------------------------------


# ===========================================================================
# [DONE] LANGKAH 1.2 - Konversi Model .pt ke .onnx (untuk Jetson Nano)
# ===========================================================================
# Penjelasan:
#   Script batch mengkonversi SEMUA model .pt dari kumpulan_model_ready/
#   ke format ONNX dalam sekali jalan.
#
#   Input  : kumpulan_model_ready/ (best_yolov8n.pt, s, m, l, x)
#   Output : jetson_nano/onnx_model/ (best_yolov8n.onnx, s, m, l, x)   [OK]
# ---------------------------------------------------------------------------


# ===========================================================================
# [DONE] LANGKAH 1.3 - Konversi Model .pt ke .tflite (untuk Raspberry Pi 4)
# ===========================================================================
# Penjelasan:
#   Script batch mengkonversi SEMUA model .pt dari kumpulan_model_ready/
#   ke format TensorFlow Lite dalam sekali jalan.
#
#   Input  : kumpulan_model_ready/ (best_yolov8n.pt, s, m, l, x)
#   Output : raspberry_pi_4/tflite_model/ (best_yolov8n_fp32.tflite, s, m, l, x) [OK]
# ---------------------------------------------------------------------------


# ===========================================================================
# [DONE] LANGKAH 1.4 - Folder Device Sudah Self-Contained
# ===========================================================================
# Penjelasan:
#   Semua file yang dibutuhkan sudah ada di dalam folder masing-masing:
#
#   D:\RISET\drone-wisard\test_for_device\jetson_nano\
#     ├── benchmark_jetson.py        (23.5 KB)
#     ├── evaluate_accuracy.py       (17.4 KB)
#     ├── convert_onnx_to_trt.py     (15.1 KB)
#     ├── power_logger.py            (script logging daya via jtop)
#     ├── onnx_model\                (5 file .onnx, total ~580 MB)
#     │   ├── best_yolov8n.onnx      (11.7 MB)
#     │   ├── best_yolov8s.onnx      (42.7 MB)
#     │   ├── best_yolov8m.onnx      (98.8 MB)
#     │   ├── best_yolov8l.onnx      (166.6 MB)
#     │   └── best_yolov8x.onnx      (260.2 MB)
#     ├── models\                    (kosong, diisi .engine di Jetson)
#     ├── test\                      (5950 gambar, ~335 MB)
#     └── test_labels\               (5950 label)
#
#   D:\RISET\drone-wisard\test_for_device\raspberry_pi_4\
#     ├── benchmark_raspi.py         (26.0 KB)
#     ├── evaluate_accuracy.py       (17.4 KB)
#     ├── tflite_model\              (5 file .tflite, total ~580 MB)
#     │   ├── best_yolov8n_fp32.tflite  (11.7 MB)
#     │   ├── best_yolov8s_fp32.tflite  (42.7 MB)
#     │   ├── best_yolov8m_fp32.tflite  (98.8 MB)
#     │   ├── best_yolov8l_fp32.tflite  (166.6 MB)
#     │   └── best_yolov8x_fp32.tflite  (260.2 MB)
#     ├── test\                      (5950 gambar, ~335 MB)
#     └── test_labels\               (5950 label)
#
#   Tinggal copy SELURUH folder ke flashdisk.
# ---------------------------------------------------------------------------


################################################################################
#                                                                              #
#           FASE 2: TRANSFER FILE KE DEVICE (VIA FLASHDISK USB)                #
#              (Siapkan flashdisk di PC, lalu colokkan ke device)              #
#                                                                              #
################################################################################

# ===========================================================================
# [DONE]LANGKAH 2.1 - Copy Folder ke Flashdisk (di PC)
# ===========================================================================
# Penjelasan:
#   Copy SELURUH folder device ke flashdisk. Tidak perlu pilih-pilih file.
#   Masing-masing folder sudah berisi semua yang dibutuhkan.
#
#   Estimasi ukuran:
#     jetson_nano/     ~920 MB (onnx ~580 + test ~335 + script)
#     raspberry_pi_4/  ~920 MB (tflite ~580 + test ~335 + script)
#     Total:           ~1.8 GB (gunakan flashdisk minimal 4 GB)
#
#   Ganti F: dengan huruf drive flashdisk Anda.
# ---------------------------------------------------------------------------

# Copy folder Jetson Nano ke flashdisk
xcopy /E /I "D:\RISET\drone-wisard\test_for_device\jetson_nano" "F:\jetson_nano"

# Copy folder Raspberry Pi 4 ke flashdisk
xcopy /E /I "D:\RISET\drone-wisard\test_for_device\raspberry_pi_4" "F:\raspberry_pi_4"


# ===========================================================================
# LANGKAH 2.2 - Copy dari Flashdisk ke Jetson Nano
# ===========================================================================
# Penjelasan:
#   Colokkan flashdisk ke Jetson Nano.
#   Buka terminal di layar Jetson Nano (klik kanan desktop > Open Terminal).
#
#   Di Linux, flashdisk biasanya auto-mount di:
#     /media/<username>/<NAMA_FLASHDISK>
#   Jika tidak auto-mount:
#     sudo mount /dev/sda1 /mnt/usb
# ---------------------------------------------------------------------------

# Ketik di terminal Jetson Nano:

# Cek mount point flashdisk
lsblk
ls /media/$USER/

# [DONE] Copy seluruh folder dari flashdisk ke home directory
# [DONE] (ganti <FLASHDISK> dengan nama flashdisk Anda)
cp -r /media/$USER/<FLASHDISK>/jetson_nano ~/benchmark

# [DONE] Verifikasi
ls ~/benchmark/
ls ~/benchmark/onnx_model/*.onnx
ls ~/benchmark/test/*.jpg | wc -l
ls ~/benchmark/test_labels/*.txt | wc -l


# ===========================================================================
# LANGKAH 2.3 - Copy dari Flashdisk ke Raspberry Pi 4
# ===========================================================================
# Penjelasan:
#   Colokkan flashdisk ke Raspberry Pi 4.
#   Buka terminal di layar RPi4 (klik kanan desktop > Open Terminal,
#   atau tekan Ctrl+Alt+T).
# ---------------------------------------------------------------------------

# Ketik di terminal RPi4:

# Cek mount point flashdisk
lsblk
ls /media/$USER/

# Copy seluruh folder dari flashdisk ke home directory
cp -r /media/$USER/<FLASHDISK>/raspberry_pi_4 ~/benchmark

# Verifikasi
ls ~/benchmark/
ls ~/benchmark/tflite_model/*.tflite
ls ~/benchmark/test/*.jpg | wc -l
ls ~/benchmark/test_labels/*.txt | wc -l


################################################################################
#                                                                              #
#                   FASE 3: EKSEKUSI DI NVIDIA JETSON NANO                     #
#       (Langsung di layar/monitor Jetson Nano, buka Terminal / cmd)           #
#                                                                              #
################################################################################

# ===========================================================================
# LANGKAH 3.1 - Optimasi Performa Jetson Nano (WAJIB, sekali saja)
# ===========================================================================
# Penjelasan:
#   Buka terminal di layar Jetson Nano, lalu jalankan perintah berikut.
#   Mengatur Jetson Nano ke mode performa maksimal:
#   - nvpmodel -m 0  = mode 10W (4 core aktif, clock maksimal)
#   - jetson_clocks  = kunci frekuensi CPU/GPU/EMC ke nilai tertinggi
#
#   PENTING: Jika tidak dilakukan, Jetson berjalan dalam mode hemat daya
#   dan hasil benchmark akan jauh lebih lambat dari seharusnya!
# ---------------------------------------------------------------------------

sudo nvpmodel -m 0
sudo jetson_clocks

# Verifikasi mode sudah benar:
sudo nvpmodel -q


# ===========================================================================
# LANGKAH 3.2 - Install Dependencies (sekali saja)
# ===========================================================================
# Penjelasan:
#   Install library Python yang dibutuhkan oleh script benchmark.
#   - ultralytics : framework YOLOv8
#   - opencv-python-headless : pemrosesan gambar tanpa GUI
#   - psutil : monitoring CPU/RAM
#   - numpy : komputasi numerik
#   - jetson-stats : monitoring daya/power via jtop (untuk power_logger.py)
#
#   PENTING untuk jetson-stats:
#     Setelah install, restart service & tambahkan user ke group jtop:
#       sudo systemctl restart jtop.service
#       sudo usermod -aG jtop $USER
#     Lalu LOGOUT & LOGIN ULANG agar group aktif.
# ---------------------------------------------------------------------------

pip3 install ultralytics opencv-python-headless psutil numpy
sudo pip3 install jetson-stats
sudo systemctl restart jtop.service
sudo usermod -aG jtop $USER
# ^^^ Setelah perintah di atas, LOGOUT & LOGIN ULANG sebelum lanjut


# ===========================================================================
# LANGKAH 3.3 - Konversi ONNX ke TensorRT .engine (di Jetson Nano)
# ===========================================================================
# Penjelasan:
#   Konversi model dari format ONNX ke TensorRT (.engine).
#   HARUS dilakukan di Jetson Nano karena TensorRT engine bersifat
#   platform-specific (engine yang dibuat di PC tidak bisa jalan di Jetson).
#
#   --fp16 mengaktifkan presisi FP16 yang 2x lebih cepat di GPU Maxwell
#   tanpa penurunan akurasi yang signifikan.
#
#   Proses ini memakan waktu 5-30 menit per model tergantung ukurannya.
#   Jalankan untuk SETIAP file .onnx satu per satu.
#
#   Hasil .engine otomatis disimpan di folder ~/benchmark/models/
# ---------------------------------------------------------------------------

cd ~/benchmark

# [DONE] ---- YOLOv8n ----
python3 convert_onnx_to_trt.py --input onnx_model/best_yolov8n.onnx --fp16

# [DONE] ---- YOLOv8s ----
python3 convert_onnx_to_trt.py --input onnx_model/best_yolov8s.onnx --fp16

# [DONE] ---- YOLOv8m ----
python3 convert_onnx_to_trt.py --input onnx_model/best_yolov8m.onnx --fp16

# [DONE] ---- YOLOv8l ----
python3 convert_onnx_to_trt.py --input onnx_model/best_yolov8l.onnx --fp16

# [DONE] ---- YOLOv8x ----
python3 convert_onnx_to_trt.py --input onnx_model/best_yolov8x.onnx --fp16

# [DONE] Verifikasi semua engine sudah ada
ls -la models/*.engine


# ===========================================================================
# LANGKAH 3.4 - Benchmark FPS & Latency + Power Logging
#               (SEMUA Model, satu per satu)
# ===========================================================================
# Penjelasan:
#   Menjalankan inferensi pada 5950 gambar test dan mencatat:
#   - FPS (Frames Per Second)
#   - Latency: Preprocess, Inference, Postprocess, Total (ms)
#   - Resource: CPU%, GPU%, RAM (MB), Suhu (C) [background thread]
#   - DAYA/POWER: Total, GPU, CPU (mW) via jtop [terminal terpisah]
#
#   Jalankan SATU PER SATU agar resource monitoring akurat.
#   Setelah satu selesai, baru jalankan model berikutnya.
#
#   *** METODE: 2 TERMINAL ***
#   Terminal 1 = Power Logger (jalankan DULUAN, stop dengan Ctrl+C)
#   Terminal 2 = Benchmark (jalankan SETELAH power logger aktif)
#
#   Data output mengisi tabel_data.xlsx:
#   ┌─────────────────────────────────────────────────────────────────┐
#   │ Sheet "fps&latency ke2perangkat" (Tabel 4.2):                  │
#   │   Preprocess, Inferensi, Postprocess, Total, FPS, Status       │
#   │   <-- dari benchmark_summary_*.json field latency.*.mean       │
#   │                                                                │
#   │ Sheet "Penggunaan resource perangkat" (Tabel 4.4):             │
#   │   CPU Avg/Max, GPU Avg/Max, RAM Avg/Max/Total, Suhu Avg/Max   │
#   │   <-- dari benchmark_resource_*.csv (agregasi per kolom)       │
#   │                                                                │
#   │ Data Daya/Power (Tabel tambahan):                              │
#   │   Total, GPU, CPU power (mW), suhu, utilisasi                  │
#   │   <-- dari power_log_*.csv (per detik via jtop)                │
#   └─────────────────────────────────────────────────────────────────┘
#
#   Output file per model (disimpan di ~/benchmark/):
#   - benchmark_fps_latency_{model}_jetson_nano.csv
#   - benchmark_resource_{model}_jetson_nano.csv
#   - benchmark_summary_{model}_jetson_nano.json
#   - power_log_{model}_jetson_nano.csv              <-- BARU (dari jtop)
# ---------------------------------------------------------------------------

cd ~/benchmark

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ALUR PER MODEL:                                                       ║
# ║  1. Terminal 1: jalankan power_logger.py --label <model>               ║
# ║  2. Terminal 2: jalankan benchmark_jetson.py --model <model>           ║
# ║  3. Tunggu benchmark selesai                                           ║
# ║  4. Terminal 1: tekan Ctrl+C untuk stop power logger                   ║
# ║  5. Ulangi untuk model berikutnya                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ---- YOLOv8n ----
# [Terminal 1] Power logger (jalankan DULUAN, biarkan berjalan):
python3 power_logger.py --label yolov8n
# [Terminal 2] Benchmark:
python3 benchmark_jetson.py \
    --model models/best_yolov8n_fp16.engine \
    --images test/ \
    --warmup 20 --conf 0.25 --imgsz 640
# [Terminal 1] Setelah benchmark selesai → Ctrl+C

# ---- YOLOv8s ----
# [Terminal 1]:
python3 power_logger.py --label yolov8s
# [Terminal 2]:
python3 benchmark_jetson.py \
    --model models/best_yolov8s_fp16.engine \
    --images test/ \
    --warmup 20 --conf 0.25 --imgsz 640
# [Terminal 1] → Ctrl+C

# ---- YOLOv8m ----
# [Terminal 1]:
python3 power_logger.py --label yolov8m
# [Terminal 2]:
python3 benchmark_jetson.py \
    --model models/best_yolov8m_fp16.engine \
    --images test/ \
    --warmup 20 --conf 0.25 --imgsz 640
# [Terminal 1] → Ctrl+C

# ---- YOLOv8l ----
# [Terminal 1]:
python3 power_logger.py --label yolov8l
# [Terminal 2]:
python3 benchmark_jetson.py \
    --model models/best_yolov8l_fp16.engine \
    --images test/ \
    --warmup 20 --conf 0.25 --imgsz 640
# [Terminal 1] → Ctrl+C

# ---- YOLOv8x ----
# [Terminal 1]:
python3 power_logger.py --label yolov8x
# [Terminal 2]:
python3 benchmark_jetson.py \
    --model models/best_yolov8x_fp16.engine \
    --images test/ \
    --warmup 20 --conf 0.25 --imgsz 640
# [Terminal 1] → Ctrl+C


# ===========================================================================
# LANGKAH 3.5 - Evaluasi Akurasi Model (SEMUA Model, satu per satu)
# ===========================================================================
# Penjelasan:
#   Mengevaluasi seberapa akurat model setelah dikonversi ke TensorRT.
#   Membandingkan prediksi model dengan ground truth label.
#
#   Data output mengisi tabel_data.xlsx:
#   ┌─────────────────────────────────────────────────────────────────┐
#   │ Sheet "eval akurasi setelah convert" (Tabel 4.3):              │
#   │   Kolom "Jetson Nano (.engine FP16)":                          │
#   │   Precision, Recall, F1, mAP50                                 │
#   │   <-- dari eval_accuracy_*.json field metrics.*                 │
#   └─────────────────────────────────────────────────────────────────┘
#
#   Output file per model:
#   - eval_accuracy_{model}_jetson_nano.json
# ---------------------------------------------------------------------------

cd ~/benchmark

#  [DONE]  ---- YOLOv8n ----
python3 evaluate_accuracy.py \
    --model models/best_yolov8n_fp16.engine \
    --images test/ --labels test_labels/ \
    --device 0 --device-name jetson_nano \
    --conf 0.25 --iou 0.5

#  [DONE] ---- YOLOv8s ----
python3 evaluate_accuracy.py \
    --model models/best_yolov8s_fp16.engine \
    --images test/ --labels test_labels/ \
    --device 0 --device-name jetson_nano \
    --conf 0.25 --iou 0.5

#  [DONE]  ---- YOLOv8m ----
python3 evaluate_accuracy.py \
    --model models/best_yolov8m_fp16.engine \
    --images test/ --labels test_labels/ \
    --device 0 --device-name jetson_nano \
    --conf 0.25 --iou 0.5

#  [DONE]  ---- YOLOv8l ----
python3 evaluate_accuracy.py \
    --model models/best_yolov8l_fp16.engine \
    --images test/ --labels test_labels/ \
    --device 0 --device-name jetson_nano \
    --conf 0.25 --iou 0.5

# [DONE]---- YOLOv8x ----
python3 evaluate_accuracy.py \
    --model models/best_yolov8x_fp16.engine \
    --images test/ --labels test_labels/ \
    --device 0 --device-name jetson_nano \
    --conf 0.25 --iou 0.5


# ===========================================================================
# LANGKAH 3.6 - Verifikasi & Salin Hasil ke Flashdisk
# ===========================================================================
# Penjelasan:
#   Cek semua file output, lalu salin ke flashdisk untuk dibawa ke PC.
#
#   File yang harus ada (per model x 5 model = total 25 file):
#     benchmark_fps_latency_{model}_jetson_nano.csv    --> Tabel 4.2
#     benchmark_resource_{model}_jetson_nano.csv       --> Tabel 4.4
#     benchmark_summary_{model}_jetson_nano.json       --> Tabel 4.2
#     eval_accuracy_{model}_jetson_nano.json           --> Tabel 4.3
#     power_log_{model}_jetson_nano.csv                --> Data daya/power
# ---------------------------------------------------------------------------

cd ~/benchmark

# Lihat semua file hasil
ls -la *.csv *.json

# Colokkan flashdisk, lalu salin hasil ke flashdisk
# (ganti <FLASHDISK> dengan nama flashdisk Anda)
mkdir -p /media/$USER/<FLASHDISK>/hasil_jetson
cp ~/benchmark/*.csv  /media/$USER/<FLASHDISK>/hasil_jetson/
cp ~/benchmark/*.json /media/$USER/<FLASHDISK>/hasil_jetson/

# Verifikasi
ls /media/$USER/<FLASHDISK>/hasil_jetson/


################################################################################
#                                                                              #
#                   FASE 4: EKSEKUSI DI RASPBERRY PI 4                         #
#        (Langsung di layar/monitor RPi4, buka Terminal / cmd)                 #
#                                                                              #
#   PILIH SALAH SATU OPSI:                                                     #
#     OPSI A: TFLite (.tflite) + Python 3.11  [RECOMMENDED - performa terbaik] #
#     OPSI B: ONNX (.onnx) + Python 3.13     [jika tidak bisa install Py 3.11]#
#                                                                              #
#   Kedua opsi menghasilkan output format IDENTIK untuk generate_excel.py.     #
#                                                                              #
################################################################################

# ===========================================================================
# PERBANDINGAN OPSI A vs OPSI B
# ===========================================================================
#
#   +-----------------+-----------------------------+-----------------------------+
#   |                 | OPSI A: TFLite              | OPSI B: ONNX                |
#   +-----------------+-----------------------------+-----------------------------+
#   | Format model    | .tflite (FP32)              | .onnx                       |
#   | Python          | 3.11 (perlu install)        | 3.13 (sudah ada)            |
#   | Runtime         | tflite-runtime + XNNPACK    | onnxruntime                 |
#   | Performa ARM    | LEBIH BAIK (XNNPACK/NEON)   | Standar                     |
#   | Script folder   | opsi_tflite/                | opsi_onnx/                  |
#   | Model folder    | tflite_model/               | onnx_model/                 |
#   | Output format   | IDENTIK                     | IDENTIK                     |
#   +-----------------+-----------------------------+-----------------------------+
#


################################################################################
#                                                                              #
#     OPSI A: TFLite + Python 3.11 (RECOMMENDED - performa terbaik)            #
#                                                                              #
################################################################################

# ===========================================================================
# LANGKAH A.1 - Buat virtualenv Python 3.11.9 (pyenv) + install deps
# ===========================================================================
# Penjelasan:
#   tflite-runtime hanya tersedia hingga Python 3.11.
#   RPi4 Anda sudah punya Python 3.11.9 via pyenv.
#   Buat virtualenv dengan Python 3.11.9, lalu install dependencies.
#   Proses pip install bisa 10-20 menit di RPi4.
# ---------------------------------------------------------------------------

~/.pyenv/versions/3.11.9/bin/python -m venv ~/tflite_env
source ~/tflite_env/bin/activate

# Verifikasi versi Python
python --version
# Harus menampilkan: Python 3.11.9

pip install --upgrade pip
pip install ultralytics opencv-python-headless psutil numpy tflite-runtime

# Verifikasi tflite-runtime
python -c "import tflite_runtime; print('tflite-runtime OK')"


# ===========================================================================
# LANGKAH A.3 - Copy script TFLite ke benchmark folder
# ===========================================================================
# Penjelasan:
#   Copy script dari opsi_tflite/ ke ~/benchmark/ (working directory).
#   Script benchmark dan evaluate sudah disesuaikan untuk format TFLite.
# ---------------------------------------------------------------------------

cd ~/benchmark
cp opsi_tflite/benchmark_raspi.py .
cp opsi_tflite/evaluate_accuracy.py .


# ===========================================================================
# LANGKAH A.4 - Benchmark FPS & Latency TFLite (satu per satu)
# ===========================================================================
# Penjelasan:
#   Jalankan benchmark menggunakan model .tflite.
#   TFLite menggunakan XNNPACK delegate yang dioptimasi untuk ARM NEON,
#   sehingga performa LEBIH BAIK dibanding ONNX Runtime di ARM.
#
#   PERINGATAN untuk model besar:
#   - YOLOv8l dan YOLOv8x KEMUNGKINAN BESAR OUT OF MEMORY (OOM)
#   - RPi4 hanya punya 4GB RAM, model besar butuh lebih dari itu
#   - Jika OOM terjadi, script otomatis mencatatnya (tidak crash)
#
#   Output file per model (disimpan di ~/benchmark/):
#   - benchmark_fps_latency_{model}_raspi4.csv    -> per-frame latency
#   - benchmark_resource_{model}_raspi4.csv       -> CPU/RAM/Suhu per detik
#   - benchmark_summary_{model}_raspi4.json       -> ringkasan statistik
# ---------------------------------------------------------------------------

cd ~/benchmark
source ~/tflite_env/bin/activate

# [DONE]---- YOLOv8n ----
python3 benchmark_raspi.py \
    --model tflite_model/best_yolov8n_fp32.tflite \
    --images test/ \
    --warmup 10 --conf 0.25 --imgsz 640

# [DONE] ---- YOLOv8s ----
python3 benchmark_raspi.py \
    --model tflite_model/best_yolov8s_fp32.tflite \
    --images test/ \
    --warmup 10 --conf 0.25 --imgsz 640

# [DONE] ---- YOLOv8m ----
python3 benchmark_raspi.py \
    --model tflite_model/best_yolov8m_fp32.tflite \
    --images test/ \
    --warmup 10 --conf 0.25 --imgsz 640

# [DONE]---- YOLOv8l (MUNGKIN OOM!) ----
python3 benchmark_raspi.py \
    --model tflite_model/best_yolov8l_fp32.tflite \
    --images test/ \
    --warmup 10 --conf 0.25 --imgsz 640

# [DONE]---- YOLOv8x (MUNGKIN OOM!) ----
python3 benchmark_raspi.py \
    --model tflite_model/best_yolov8x_fp32.tflite \
    --images test/ \
    --warmup 10 --conf 0.25 --imgsz 640


# ===========================================================================
# LANGKAH A.5 - Evaluasi Akurasi TFLite (satu per satu)
# ===========================================================================
# Penjelasan:
#   Evaluasi akurasi model TFLite: Precision, Recall, F1, mAP50.
#
#   Output: eval_accuracy_{model}_raspi4.json
# ---------------------------------------------------------------------------

cd ~/benchmark
source ~/tflite_env/bin/activate

# [DONE] ---- YOLOv8n ----
python3 evaluate_accuracy.py \
    --model tflite_model/best_yolov8n_fp32.tflite \
    --images test/ --labels test_labels/ \
    --device cpu --device-name raspi4 \
    --conf 0.25 --iou 0.5

# [DONE] ---- YOLOv8s ----
python3 evaluate_accuracy.py \
    --model tflite_model/best_yolov8s_fp32.tflite \
    --images test/ --labels test_labels/ \
    --device cpu --device-name raspi4 \
    --conf 0.25 --iou 0.5

# [DONE] ---- YOLOv8m ----
python3 evaluate_accuracy.py \
    --model tflite_model/best_yolov8m_fp32.tflite \
    --images test/ --labels test_labels/ \
    --device cpu --device-name raspi4 \
    --conf 0.25 --iou 0.5

# [DONE] ---- YOLOv8l ----
python3 evaluate_accuracy.py \
    --model tflite_model/best_yolov8l_fp32.tflite \
    --images test/ --labels test_labels/ \
    --device cpu --device-name raspi4 \
    --conf 0.25 --iou 0.5

# ---- YOLOv8x ------
python3 evaluate_accuracy.py \
    --model tflite_model/best_yolov8x_fp32.tflite \
    --images test/ --labels test_labels/ \
    --device cpu --device-name raspi4 \
    --conf 0.25 --iou 0.5


# ===========================================================================
# LANGKAH A.6 - Verifikasi & Salin Hasil ke Flashdisk
# ===========================================================================

cd ~/benchmark
ls -la *.csv *.json

mkdir -p /media/$USER/<FLASHDISK>/hasil_raspi4
cp ~/benchmark/*.csv  /media/$USER/<FLASHDISK>/hasil_raspi4/
cp ~/benchmark/*.json /media/$USER/<FLASHDISK>/hasil_raspi4/

ls /media/$USER/<FLASHDISK>/hasil_raspi4/


################################################################################
#                                                                              #
#    tidak dipakai OPSI B: ONNX + Python 3.13 (alternatif jika tidak bisa install Py 3.11) #
#                                                                              #
################################################################################

# ===========================================================================
# LANGKAH B.1 - Buat virtualenv Python 3.13 + install deps
# ===========================================================================
# Penjelasan:
#   Gunakan Python 3.13 yang sudah ada di RPi4 Anda.
#   Install onnxruntime sebagai pengganti tflite-runtime.
#   Performa sedikit lebih lambat dibanding TFLite XNNPACK di ARM.
# ---------------------------------------------------------------------------

python3 -m venv ~/onnx_env
source ~/onnx_env/bin/activate

pip install --upgrade pip
pip install ultralytics opencv-python-headless psutil numpy onnxruntime


# ===========================================================================
# LANGKAH B.2 - Copy script ONNX ke benchmark folder
# ===========================================================================

cd ~/benchmark
cp opsi_onnx/benchmark_raspi.py .
cp opsi_onnx/evaluate_accuracy.py .


# ===========================================================================
# LANGKAH B.3 - Benchmark FPS & Latency ONNX (satu per satu)
# ===========================================================================
# Penjelasan:
#   Sama seperti OPSI A tapi menggunakan model .onnx.
#   Output file formatnya IDENTIK (bisa diproses oleh generate_excel.py).
# ---------------------------------------------------------------------------

cd ~/benchmark
source ~/onnx_env/bin/activate

# ---- YOLOv8n ----
python3 benchmark_raspi.py \
    --model onnx_model/best_yolov8n.onnx \
    --images test/ \
    --warmup 10 --conf 0.25 --imgsz 640

# ---- YOLOv8s ----
python3 benchmark_raspi.py \
    --model onnx_model/best_yolov8s.onnx \
    --images test/ \
    --warmup 10 --conf 0.25 --imgsz 640

# ---- YOLOv8m ----
python3 benchmark_raspi.py \
    --model onnx_model/best_yolov8m.onnx \
    --images test/ \
    --warmup 10 --conf 0.25 --imgsz 640

# ---- YOLOv8l (MUNGKIN OOM!) ----
python3 benchmark_raspi.py \
    --model onnx_model/best_yolov8l.onnx \
    --images test/ \
    --warmup 10 --conf 0.25 --imgsz 640

# ---- YOLOv8x (MUNGKIN OOM!) ----
python3 benchmark_raspi.py \
    --model onnx_model/best_yolov8x.onnx \
    --images test/ \
    --warmup 10 --conf 0.25 --imgsz 640


# ===========================================================================
# LANGKAH B.4 - Evaluasi Akurasi ONNX (satu per satu)
# ===========================================================================

cd ~/benchmark
source ~/onnx_env/bin/activate

# ---- YOLOv8n ----
python3 evaluate_accuracy.py \
    --model onnx_model/best_yolov8n.onnx \
    --images test/ --labels test_labels/ \
    --device cpu --device-name raspi4 \
    --conf 0.25 --iou 0.5

# ---- YOLOv8s ----
python3 evaluate_accuracy.py \
    --model onnx_model/best_yolov8s.onnx \
    --images test/ --labels test_labels/ \
    --device cpu --device-name raspi4 \
    --conf 0.25 --iou 0.5

# ---- YOLOv8m ----
python3 evaluate_accuracy.py \
    --model onnx_model/best_yolov8m.onnx \
    --images test/ --labels test_labels/ \
    --device cpu --device-name raspi4 \
    --conf 0.25 --iou 0.5

# ---- YOLOv8l (MUNGKIN OOM!) ----
python3 evaluate_accuracy.py \
    --model onnx_model/best_yolov8l.onnx \
    --images test/ --labels test_labels/ \
    --device cpu --device-name raspi4 \
    --conf 0.25 --iou 0.5

# ---- YOLOv8x (MUNGKIN OOM!) ----
python3 evaluate_accuracy.py \
    --model onnx_model/best_yolov8x.onnx \
    --images test/ --labels test_labels/ \
    --device cpu --device-name raspi4 \
    --conf 0.25 --iou 0.5


# ===========================================================================
# LANGKAH B.5 - Verifikasi & Salin Hasil ke Flashdisk
# ===========================================================================

cd ~/benchmark
ls -la *.csv *.json

mkdir -p /media/$USER/<FLASHDISK>/hasil_raspi4
cp ~/benchmark/*.csv  /media/$USER/<FLASHDISK>/hasil_raspi4/
cp ~/benchmark/*.json /media/$USER/<FLASHDISK>/hasil_raspi4/

ls /media/$USER/<FLASHDISK>/hasil_raspi4/



################################################################################
#                                                                              #
#           FASE 5: SALIN HASIL DARI FLASHDISK KE PC                           #
#          (Colokkan flashdisk ke PC, salin file via Explorer/cmd)             #
#                                                                              #
################################################################################

# ===========================================================================
# LANGKAH 5.1 - Salin Hasil Jetson Nano dari Flashdisk ke PC
# ===========================================================================
# Penjelasan:
#   Colokkan flashdisk ke PC, lalu salin file CSV dan JSON ke folder
#   jetson_nano\ di PC.
#   File-file ini akan dibaca untuk mengisi tabel_data.xlsx.
#
#   Ganti F: dengan huruf drive flashdisk Anda.
# ---------------------------------------------------------------------------

copy F:\hasil_jetson\*.csv  D:\RISET\drone-wisard\test_for_device\jetson_nano\
copy F:\hasil_jetson\*.json D:\RISET\drone-wisard\test_for_device\jetson_nano\


# ===========================================================================
# LANGKAH 5.2 - Salin Hasil Raspberry Pi 4 dari Flashdisk ke PC
# ===========================================================================

copy F:\hasil_raspi4\*.csv  D:\RISET\drone-wisard\test_for_device\raspberry_pi_4\
copy F:\hasil_raspi4\*.json D:\RISET\drone-wisard\test_for_device\raspberry_pi_4\


# ===========================================================================
# LANGKAH 5.3 - Verifikasi File yang Tersalin
# ===========================================================================
# Penjelasan:
#   Pastikan semua file sudah tersalin dengan benar.
#   Harus ada 4 file per model x 5 model = 20 file per device.
# ---------------------------------------------------------------------------

# Cek file Jetson Nano
dir D:\RISET\drone-wisard\test_for_device\jetson_nano\*.csv
dir D:\RISET\drone-wisard\test_for_device\jetson_nano\*.json

# Cek file Raspberry Pi 4
dir D:\RISET\drone-wisard\test_for_device\raspberry_pi_4\*.csv
dir D:\RISET\drone-wisard\test_for_device\raspberry_pi_4\*.json


################################################################################
#                                                                              #
#              FASE 6: ANALISIS & GENERATE TABEL DI PC                         #
#            (Dijalankan di PC - Command Prompt / PowerShell)                   #
#                                                                              #
################################################################################

# ===========================================================================
# LANGKAH 6.1 - Update tabel_data.xlsx dari Hasil Benchmark
# ===========================================================================
# Penjelasan:
#   Script membaca semua file JSON/CSV dari folder jetson_nano\ dan
#   raspberry_pi_4\, lalu mengisi data ke sheet-sheet tabel_data.xlsx:
#
#   Input files --> Sheet yang diisi:
#   ┌──────────────────────────────────────────────────────────────────┐
#   │ benchmark_summary_*.json  --> "fps&latency ke2perangkat"        │
#   │   field: latency.preprocess_ms.mean                             │
#   │          latency.inference_ms.mean                               │
#   │          latency.postprocess_ms.mean                             │
#   │          latency.total_ms.mean                                   │
#   │          fps.mean                                                │
#   │          status                                                  │
#   ├──────────────────────────────────────────────────────────────────┤
#   │ eval_accuracy_*.json      --> "eval akurasi setelah convert"    │
#   │   field: metrics.precision                                       │
#   │          metrics.recall                                          │
#   │          metrics.f1_score                                        │
#   │          metrics.mAP50                                           │
#   ├──────────────────────────────────────────────────────────────────┤
#   │ benchmark_resource_*.csv  --> "Penggunaan resource perangkat"   │
#   │   kolom: cpu_percent (avg, max)                                  │
#   │          gpu_percent (avg, max)                                  │
#   │          ram_used_mb (avg, max)                                  │
#   │          ram_total_mb                                            │
#   │          temperature_c (avg, max)                                │
#   └──────────────────────────────────────────────────────────────────┘
#
#   Bisa dijalankan BERULANG KALI setiap kali ada data baru.
# ---------------------------------------------------------------------------

cd D:\RISET\drone-wisard\test_for_device
python generate_excel.py --update


# ===========================================================================
# LANGKAH 6.2 - Buka Hasil
# ===========================================================================

# Buka Excel untuk verifikasi
start D:\RISET\drone-wisard\penulisan\tabel_data.xlsx


################################################################################
#                                                                              #
#            MAPPING: FILE OUTPUT --> KOLOM TABEL_DATA.XLSX                     #
#                                                                              #
################################################################################
#
# ┌────────────────────────────────────┬────────────────────────────────────┐
# │ Sheet: "Performa Model YOLOv8"     │ Sumber: train_result/results.csv  │
# │ (Tabel 4.1 - Baseline PC)          │ Script: 01.generate_tabel_...     │
# │                                    │ Data: Presisi, Recall, F1, mAP50  │
# │                                    │       mAP50-95, Epoch info        │
# ├────────────────────────────────────┼────────────────────────────────────┤
# │ Sheet: "fps&latency ke2perangkat"  │ Sumber: benchmark_summary_*.json  │
# │ (Tabel 4.2)                        │ Jetson: latency.*.mean, fps.mean  │
# │                                    │ RPi4:   latency.*.mean, fps.mean  │
# ├────────────────────────────────────┼────────────────────────────────────┤
# │ Sheet: "eval akurasi setelah       │ Sumber: eval_accuracy_*.json      │
# │ convert" (Tabel 4.3)               │ Jetson: metrics.precision, etc.   │
# │                                    │ RPi4:   metrics.precision, etc.   │
# │                                    │ Penurunan: baseline - device      │
# ├────────────────────────────────────┼────────────────────────────────────┤
# │ Sheet: "Penggunaan resource        │ Sumber: benchmark_resource_*.csv  │
# │ perangkat" (Tabel 4.4)             │ Jetson: CPU,GPU,RAM,Suhu avg/max  │
# │                                    │ RPi4:   CPU,RAM,Suhu avg/max      │
# │                                    │         (GPU = N/A)               │
# └────────────────────────────────────┴────────────────────────────────────┘
#
#
################################################################################
#                                                                              #
#                    RINGKASAN FILE PER DEVICE                                 #
#                                                                              #
################################################################################
#
# === FILE YANG DIJALANKAN DI PC ===
#
#   1. strip_optimizer.py             - Hapus optimizer state dari best.pt
#   2. 02.batch_convert_to_onnx.py    - Konversi .pt -> .onnx (5 model)
#   3. 03.batch_convert_to_tflite.py  - Konversi .pt -> .tflite (5 model)
#   4. 01.generate_tabel_baseline.py  - Buat Tabel 4.1 dari log training
#   5. generate_excel.py              - Update tabel_data.xlsx dari hasil
#
# === FILE DI FOLDER jetson_nano/ (dijalankan di Jetson Nano) ===
#
#   1. convert_onnx_to_trt.py   - Konversi .onnx -> .engine (TensorRT)
#   2. benchmark_jetson.py      - Ukur FPS, latency, CPU/GPU/RAM/suhu
#   3. evaluate_accuracy.py     - Ukur Precision, Recall, F1, mAP50
#   4. power_logger.py          - Logging daya/power via jtop ke CSV
#   + onnx_model/               - 5 file .onnx (input konversi)
#   + models/                   - output .engine (hasil konversi)
#   + test/                     - 5950 gambar test
#   + test_labels/              - 5950 label ground truth
#
# === FILE DI FOLDER raspberry_pi_4/ (dijalankan di RPi4) ===
#
#   1. benchmark_raspi.py       - Ukur FPS, latency, CPU/RAM/suhu
#   2. evaluate_accuracy.py     - Ukur Precision, Recall, F1, mAP50
#   + tflite_model/             - 5 file .tflite (sudah siap pakai)
#   + test/                     - 5950 gambar test
#   + test_labels/              - 5950 label ground truth
#
################################################################################
