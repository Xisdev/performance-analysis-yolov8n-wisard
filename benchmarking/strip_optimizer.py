"""
==============================================================================
  Strip Optimizer State dari Model YOLOv8 best.pt
==============================================================================
  Script ini menghapus data training (optimizer state, EMA state, gradien)
  dari file best.pt sehingga ukurannya menjadi standar.

  PENTING: Ini TIDAK mengubah kualitas/akurasi model.
  Yang dihapus hanya data untuk melanjutkan training (resume),
  bukan weights model itu sendiri.

  Hasil:
    - Ukuran file menjadi standar (sesuai ukuran resmi Ultralytics)
    - Model siap deploy ke device (Jetson Nano, RPi4, dll)
    - Bisa langsung dikonversi ke ONNX/TFLite/TensorRT

  Cara pakai:
    python strip_optimizer.py
==============================================================================
"""

import os
import sys
import shutil

# Coba import torch
try:
    import torch
except ImportError:
    print("ERROR: PyTorch tidak terinstall!")
    print("Install dulu: pip install torch")
    sys.exit(1)

# Coba import ultralytics (opsional, untuk strip yang lebih bersih)
try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path input: folder train_result
TRAIN_RESULT_DIR = os.path.join(SCRIPT_DIR, 'train_result')

# Path output: folder model siap deploy
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'kumpulan_model_ready')

# Mapping varian ke folder
VARIANTS = {
    'yolov8n': 'yolov8n_wisard_ir',
    'yolov8s': 'yolov8s_wisard_ir',
    'yolov8m': 'yolov8m_wisard_ir',
    'yolov8l': 'yolov8l_wisard_ir',
    'yolov8x': 'yolov8x_wisard_ir',
}

# Ukuran standar best.pt (setelah strip, dalam MB)
STANDARD_SIZES = {
    'yolov8n': 6.2,
    'yolov8s': 21.5,
    'yolov8m': 49.7,
    'yolov8l': 83.7,
    'yolov8x': 130.5,
}


def strip_optimizer_manual(src_path, dst_path):
    """
    Strip optimizer state secara manual menggunakan torch.
    Menghapus: optimizer, ema, updates, train_args (training-specific).
    Mempertahankan: model weights, names, nc, yaml.
    """
    print(f"  Loading: {src_path}")
    ckpt = torch.load(src_path, map_location='cpu', weights_only=False)

    if not isinstance(ckpt, dict):
        print(f"  [WARN] Format tidak dikenal, salin apa adanya")
        shutil.copy2(src_path, dst_path)
        return

    # Ambil info sebelum strip
    epoch = ckpt.get('epoch', '?')
    best_fitness = ckpt.get('best_fitness', None)
    if isinstance(best_fitness, torch.Tensor):
        best_fitness = best_fitness.item()

    print(f"  Best epoch: {epoch}")
    if best_fitness:
        print(f"  Best fitness: {best_fitness:.6f}")

    # Gunakan EMA model jika ada (EMA biasanya lebih baik)
    model = ckpt.get('ema', ckpt.get('model', None))
    if model is None:
        print(f"  [ERROR] Tidak ada model di checkpoint!")
        return

    # Set model ke mode eval dan hapus gradien
    if hasattr(model, 'half'):
        model = model.half()  # Konversi ke FP16 untuk ukuran lebih kecil
    if hasattr(model, 'eval'):
        model.eval()

    # Hapus gradien dari semua parameter
    for p in model.parameters():
        p.requires_grad_(False)

    # Buat checkpoint baru yang bersih
    clean_ckpt = {
        'model': model,
        'epoch': epoch,
        'best_fitness': best_fitness,
        'date': ckpt.get('date', None),
        'version': ckpt.get('version', None),
        'train_args': ckpt.get('train_args', None),  # Simpan untuk referensi
    }

    # Hapus key yang None
    clean_ckpt = {k: v for k, v in clean_ckpt.items() if v is not None}

    print(f"  Saving stripped model: {dst_path}")
    torch.save(clean_ckpt, dst_path)


def strip_with_ultralytics(src_path, dst_path):
    """
    Strip menggunakan Ultralytics built-in (lebih bersih dan standar).
    """
    print(f"  Loading via Ultralytics: {src_path}")

    # Salin dulu ke dst, lalu strip in-place
    shutil.copy2(src_path, dst_path)

    # Gunakan ultralytics strip_optimizer
    from ultralytics.utils.torch_utils import strip_optimizer
    strip_optimizer(dst_path)
    print(f"  Stripped via Ultralytics: {dst_path}")


def main():
    print("=" * 65)
    print("  STRIP OPTIMIZER STATE - Siapkan Model untuk Deploy")
    print("=" * 65)

    if not os.path.exists(TRAIN_RESULT_DIR):
        print(f"\n[ERROR] Folder tidak ditemukan: {TRAIN_RESULT_DIR}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nInput : {TRAIN_RESULT_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    if HAS_ULTRALYTICS:
        print(f"Metode: Ultralytics built-in (recommended)")
    else:
        print(f"Metode: Manual torch strip")

    results = []

    for variant, folder_name in VARIANTS.items():
        src_path = os.path.join(TRAIN_RESULT_DIR, folder_name, 'weights', 'best.pt')
        dst_path = os.path.join(OUTPUT_DIR, f'best_{variant}.pt')

        print(f"\n{'-' * 65}")
        print(f"  [{variant.upper()}]")

        if not os.path.exists(src_path):
            print(f"  [SKIP] best.pt tidak ditemukan: {src_path}")
            results.append((variant, 'SKIP', 0, 0))
            continue

        src_size = os.path.getsize(src_path) / (1024 * 1024)
        expected = STANDARD_SIZES.get(variant, 0)
        ratio = src_size / expected if expected > 0 else 0

        print(f"  Source : {src_path}")
        print(f"  Ukuran : {src_size:.1f} MB (standar: ~{expected} MB, rasio: {ratio:.1f}x)")

        if ratio <= 1.3:
            # Ukuran sudah normal, cukup salin
            print(f"  [OK] Ukuran sudah standar, salin langsung")
            shutil.copy2(src_path, dst_path)
        else:
            # Perlu strip
            print(f"  [STRIP] Perlu strip optimizer state ({ratio:.1f}x terlalu besar)")
            try:
                if HAS_ULTRALYTICS:
                    strip_with_ultralytics(src_path, dst_path)
                else:
                    strip_optimizer_manual(src_path, dst_path)
            except Exception as e:
                print(f"  [ERROR] Gagal strip: {e}")
                print(f"  [FALLBACK] Salin apa adanya")
                shutil.copy2(src_path, dst_path)

        dst_size = os.path.getsize(dst_path) / (1024 * 1024)
        status = "OK" if (dst_size / expected) <= 1.3 else "LARGE"
        reduction = ((src_size - dst_size) / src_size * 100) if src_size > dst_size else 0

        print(f"  Hasil  : {dst_size:.1f} MB", end="")
        if reduction > 0:
            print(f" (berkurang {reduction:.0f}%)")
        else:
            print()

        results.append((variant, status, src_size, dst_size))

    # Ringkasan
    print(f"\n{'=' * 65}")
    print(f"  RINGKASAN")
    print(f"{'=' * 65}")
    print(f"  {'Varian':<10} {'Sebelum':>10} {'Sesudah':>10} {'Standar':>10} {'Status':>8}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

    for variant, status, src_size, dst_size in results:
        expected = STANDARD_SIZES.get(variant, 0)
        icon = "[OK]" if status == "OK" else "[!!]" if status == "LARGE" else "[--]"
        print(f"  {variant:<10} {src_size:>8.1f}MB {dst_size:>8.1f}MB {expected:>8.1f}MB {icon:>8}")

    print(f"\nModel siap deploy di: {OUTPUT_DIR}")
    print(f"\nFile yang dihasilkan:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith('.pt'):
            size = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / (1024 * 1024)
            print(f"  - {f} ({size:.1f} MB)")

    print(f"\n{'=' * 65}")


if __name__ == "__main__":
    main()
