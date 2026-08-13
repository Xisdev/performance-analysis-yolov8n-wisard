"""
==============================================================================
  Konfigurasi Training YOLOv8 - Shared Hyperparameters
==============================================================================
  File ini berisi SEMUA hyperparameter yang digunakan untuk training
  kelima varian YOLOv8 (n, s, m, l, x).

  PRINSIP KEADILAN (FAIRNESS):
    Semua parameter training IDENTIK untuk semua varian, KECUALI batch size
    yang disesuaikan agar masing-masing varian memaksimalkan VRAM RTX 3060 12GB.

    Batch size berbeda BUKAN ketidakadilan -- justru ini best practice.
    Jika dipaksakan batch=8 untuk semua varian:
      - YOLOv8n hanya pakai 3GB dari 12GB (membuang 9GB)
      - YOLOv8x mungkin OOM atau harus batch=4

    Dengan auto-tuning batch per varian, setiap model mendapatkan
    kesempatan OPTIMAL sesuai kapasitasnya.

  HYPERPARAMETER TERKUNCI (LOCKED):
    Semua parameter di TRAINING_CONFIG di bawah ini TIDAK BOLEH diubah
    per varian. Jika diubah, ubah di config.py dan SEMUA varian ikut berubah.
    Ini menjamin keadilan perbandingan antar varian.

  GPU Target: NVIDIA RTX 3060 12GB VRAM
  Dataset: WiSARD IR (31554 train / 16302 val / 1 class: human)
=============================================================================="""

# ============================================================================
#  BATCH SIZE PER VARIAN
# ============================================================================
# VRAM usage AKTUAL di RTX 3060 12GB (imgsz=640, AdamW, mosaic=1.0):
#   AdamW butuh ~2x memory optimizer state vs SGD, sehingga estimasi lebih besar.
#
#   YOLOv8n (3.2M params)   --> batch=32  (~7 GB)  OK
#   YOLOv8s (11.2M params)  --> batch=24  (~8 GB)  OK
#   YOLOv8m (25.9M params)  --> batch=16  (~9 GB)  OK
#   YOLOv8l (43.7M params)  --> batch=4   (~9 GB)  OK 
#   YOLOv8x (68.2M params)  --> batch=2   (~10 GB) OK  

BATCH_SIZE = {
    'yolov8n': 32,
    'yolov8s': 24,
    'yolov8m': 16,
    'yolov8l': 4,
    'yolov8x': 2,
}

# ============================================================================
#  HYPERPARAMETER IDENTIK UNTUK SEMUA VARIAN
# ============================================================================

TRAINING_CONFIG = {
    # --- Core Training ---
    'epochs': 400,
    'imgsz': 640,
    'patience': 100,           # Early stopping: berhenti jika 100 epoch tanpa improvement
    'device': 0,               # GPU index 0
    'workers': 4,              # 4 workers untuk menjaga GPU tetap sibuk
    'val': True,
    'amp': True,               # ON - Mixed Precision (FP16), ~1.5-2x lebih cepat
    'deterministic': False,    # OFF - deterministic=True menyebabkan SIGSEGV pada model besar (l/x)
    'cache': 'disk',           # Cache gambar ke disk agar tidak decode ulang setiap epoch

    # --- Optimizer: AdamW ---
    'optimizer': 'AdamW',      # Dari SGD ke AdamW (lebih stabil, konvergensi lebih baik)
    'lr0': 0.001,              # Initial LR (AdamW optimal: 0.001, SGD: 0.01)
    'lrf': 0.01,               # Final LR = lr0 * lrf = 0.001 * 0.01 = 0.00001
    'momentum': 0.9,           # Beta1 untuk AdamW (setara momentum di SGD)
    'weight_decay': 0.01,      # AdamW sweet spot (SGD biasanya 0.0005)
    'cos_lr': True,            # Cosine annealing LR schedule
    'warmup_epochs': 5.0,      # Warmup linear dari 0 ke lr0
    'warmup_momentum': 0.8,    # Momentum awal saat warmup
    'warmup_bias_lr': 0.01,    # LR khusus bias layer saat warmup

    # --- Regularisasi ---
    'label_smoothing': 0.1,    # Cegah overconfident predictions
    'nbs': 64,                 # Nominal batch size untuk LR scaling

    # --- Output ---
    'exist_ok': True,
    'save': True,
    'save_period': 50,         # Simpan checkpoint setiap 50 epoch
    'plots': True,

    # --- Augmentasi Khusus Infrared (IR) ---
    # HSV hue & saturation = 0 karena gambar termal tidak punya warna
    'hsv_h': 0.0,              # Tidak ada pergeseran hue (IR = grayscale)
    'hsv_s': 0.0,              # Tidak ada pergeseran saturasi
    'hsv_v': 0.15,             # Variasi brightness ringan (simulasi suhu ambient)

    # --- Augmentasi Struktural (Drone POV) ---
    'degrees': 15.0,           # Rotasi ±15° (drone miring saat terbang)
    'translate': 0.2,          # Translasi ±20% (objek di pinggir frame)
    'scale': 0.5,              # Skala 50-150% (variasi ketinggian drone)
    'shear': 0.0,              # Tidak ada shear (tidak relevan untuk aerial)
    'perspective': 0.0,        # Tidak ada perspektif distortion
    'flipud': 0.5,             # Flip vertikal 50% (drone bisa terbalik)
    'fliplr': 0.5,             # Flip horizontal 50%
    'mosaic': 1.0,             # Mosaic augmentation aktif penuh
    'close_mosaic': 20,        # Matikan mosaic 20 epoch terakhir (fine-tuning)
    'mixup': 0.1,              # MixUp ringan (blend 2 gambar)
    'copy_paste': 0.1,         # Copy-paste ringan
    'erasing': 0.4,            # Random erasing (simulasi oklusi)
}

# ============================================================================
#  PATH KONFIGURASI (Relative Paths)
# ============================================================================

import os
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)  # Repository root

DATA_YAML = os.path.join(ROOT_DIR, 'dataset', 'data.yaml')
PROJECT_DIR = os.path.join(ROOT_DIR, 'runs')
COLLECTED_DIR = os.path.join(PROJECT_DIR, 'collected_best_models')

DATASET_NAME = 'wisard_ir'


# ============================================================================
#  HELPER: SALIN WEIGHTS SETELAH TRAINING
# ============================================================================

def copy_best_model(variant):
    """
    Salin best.pt dan last.pt setelah training selesai.

    Logika penyimpanan:
    1. Ultralytics menyimpan best.pt dan last.pt di:
       runs/{variant}_wisard_ir/weights/best.pt  <-- TETAP ADA

    2. Lalu salin best.pt ke folder terpusat dengan nama jelas:
       runs/collected_best_models/best_yolov8n.pt
       runs/collected_best_models/best_yolov8s.pt
       ...
    """
    run_name = f'{variant}_{DATASET_NAME}'
    run_dir = os.path.join(PROJECT_DIR, run_name)
    weights_dir = os.path.join(run_dir, 'weights')  # Folder asli Ultralytics

    best_src = os.path.join(weights_dir, 'best.pt')

    # --- Salin best.pt ke folder terpusat ---
    os.makedirs(COLLECTED_DIR, exist_ok=True)
    best_dst = os.path.join(COLLECTED_DIR, f'best_{variant}.pt')

    if os.path.exists(best_src):
        shutil.copy2(best_src, best_dst)
        size_mb = os.path.getsize(best_dst) / (1024 * 1024)
        print(f"  [COPY] best.pt --> {best_dst} ({size_mb:.1f} MB)")
    else:
        print(f"  [WARN] best.pt tidak ditemukan di {best_src}")

    # Verifikasi: best.pt TETAP ADA di folder weights asli
    if os.path.exists(best_src):
        print(f"  [OK] best.pt tetap ada di: {best_src}")
    else:
        print(f"  [WARN] best.pt hilang dari folder asli!")

    return best_dst
