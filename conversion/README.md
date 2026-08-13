# Konversi YOLOv8n ke TensorRT untuk NVIDIA Jetson Nano

## Struktur Folder

```
konversi/
├── convert_to_onnx.py        # Step 1: PT → ONNX (jalankan di PC)
├── convert_onnx_to_trt.py    # Step 2: ONNX → TensorRT (jalankan di Jetson Nano)
├── convert_result/            # Folder output hasil konversi
│   └── .gitkeep
└── README.md                  # Dokumentasi ini
```

## Alur Konversi

```
[PC/Laptop]                        [Jetson Nano]
best.pt ──→ best.onnx ──transfer──→ best.onnx ──→ best_fp16.engine
         Step 1                              Step 2
```

> **Penting:** TensorRT engine bersifat hardware-specific. File `.engine` HARUS
> dibuat di perangkat yang akan menjalankannya (Jetson Nano).

---

## Step 1: Konversi PT → ONNX (di PC/Laptop)

### Prasyarat
```bash
pip install ultralytics onnx onnxruntime
```

### Jalankan
```bash
cd D:\RISET\drone-wisard\konversi
python convert_to_onnx.py
```

Hasil: `convert_result/best.onnx`

---

## Step 2: Transfer file ke Jetson Nano

```bash
# Via SCP
scp convert_result/best.onnx user@<jetson-nano-ip>:~/konversi/convert_result/

# Atau via USB flash drive / SD card
```

Pastikan juga meng-copy `convert_onnx_to_trt.py` ke Jetson Nano.

---

## Step 3: Konversi ONNX → TensorRT Engine (di Jetson Nano)

### Prasyarat di Jetson Nano
- JetPack SDK (sudah include TensorRT & CUDA)
- Python 3.6+
- `pip3 install pycuda` (opsional, untuk metode TensorRT API)
- `pip3 install ultralytics` (opsional, untuk metode Ultralytics)

### Jalankan
```bash
cd ~/konversi
python3 convert_onnx_to_trt.py
```

### Opsi Precision
```bash
# FP16 (default, recommended untuk Jetson Nano)
python3 convert_onnx_to_trt.py --fp16

# FP32 (lebih akurat, tapi lebih lambat)
python3 convert_onnx_to_trt.py --fp32
```

### Pilih Metode Konversi
```bash
# Auto (coba semua metode, recommended)
python3 convert_onnx_to_trt.py --method auto

# Ultralytics (paling mudah)
python3 convert_onnx_to_trt.py --method ultralytics

# TensorRT Python API (paling fleksibel)
python3 convert_onnx_to_trt.py --method tensorrt

# trtexec command-line (bawaan JetPack)
python3 convert_onnx_to_trt.py --method trtexec
```

Hasil: `convert_result/best_fp16.engine`

---

## Menggunakan Engine di Jetson Nano

```python
from ultralytics import YOLO

# Load TensorRT engine
model = YOLO("convert_result/best_fp16.engine")

# Inference
results = model.predict(source="image.jpg", conf=0.5)

# Atau dari kamera
results = model.predict(source=0, show=True)
```

---

## Spesifikasi Jetson Nano

| Spec | Value |
|------|-------|
| GPU | 128-core Maxwell |
| CPU | Quad-core ARM A57 @ 1.43 GHz |
| RAM | 4 GB LPDDR4 |
| TensorRT | Termasuk dalam JetPack |
| CUDA Cores | 128 |
| FP16 Performance | 472 GFLOPs |

### Tips Performa
- Gunakan **FP16** untuk keseimbangan kecepatan & akurasi terbaik
- Set `MAX_WORKSPACE_MB = 1024` (1GB) agar tidak kehabisan RAM
- Gunakan power mode `10W` untuk performa maksimal:
  ```bash
  sudo nvpmodel -m 0
  sudo jetson_clocks
  ```

---

## Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `ModuleNotFoundError: tensorrt` | Pastikan JetPack terinstall benar |
| `Out of memory` saat build engine | Kurangi `MAX_WORKSPACE_MB` ke 512 |
| Engine lambat | Pastikan menggunakan FP16, bukan FP32 |
| `trtexec not found` | Cek path: `/usr/src/tensorrt/bin/trtexec` |
| ONNX parse error | Pastikan opset version = 11 saat export |
