# Harris Corner Detection

Implementasi algoritma Harris Corner Detector dari scratch menggunakan Python dan NumPy tanpa menggunakan fungsi bawaan `cv2.cornerHarris`.

---

## Instalasi

### 1. Clone repository
```bash
git clone https://github.com/jelitarahma/harris-corner-detection.git
cd harris-corner-detection
```

### 2. Buat dan aktifkan virtual environment

* **Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
* **Windows (CMD):**
  ```cmd
  python -m venv venv
  venv\Scripts\activate.bat
  ```
* **Linux / macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

*(Opsional)* Install package agar perintah `harris` bisa langsung dipanggil:
```bash
pip install -e .
```

---

## Cara Menjalankan

### 1. Quick Start (Default)
Jalankan detektor pada gambar contoh bawaan (`data/building.jpg`):
```bash
python -m harris_corner.main
```
atau jika sudah `pip install -e .`:
```bash
harris
```
Semua hasil visualisasi dan array numerik akan otomatis disimpan di folder `outputs/`.

---

### 2. Deteksi pada Gambar Sendiri
Jika ingin mencoba detaksi pada gambar lain (selain, gambar bawaan project). Silahkan tentukan path gambar yang ingin diproses:
```bash
python -m harris_corner.main path/ke/gambar.jpg
```

---

### 3. Custom Parameter (CLI)
Anda bisa menyesuaikan parameter deteksi sesuai kebutuhan:

```bash
python -m harris_corner.main data/building.jpg \
  --u 5 \
  --v 5 \
  --window 5 5 \
  --k 0.04 \
  --threshold 0.01 \
  --nms-ksize 3 \
  --circle-radius 1 \
  --output-dir outputs
```

**Daftar argumen:**
| Argumen | Default | Keterangan |
|---|---|---|
| `filename` | `data/building.jpg` | Path file gambar input |
| `--u` | `5` | Offset pergeseran horizontal ($x$) |
| `--v` | `5` | Offset pergeseran vertikal ($y$) |
| `--window` | `5 5` | Ukuran window lokal ($H \times W$) |
| `--k` | `0.04` | Parameter sensitivitas Harris ($0.04 - 0.06$) |
| `--threshold` | `0.01` | Ambang batas deteksi relatif terhadap respons puncak |
| `--nms-ksize` | `3` | Kernel Non-Maximum Suppression (`0` untuk mematikan NMS) |
| `--circle-radius` | `1` | Ukuran marker titik sudut pada gambar output |
| `--output-dir` | `outputs` | Folder tujuan penyimpanan hasil |

Untuk melihat daftar lengkap argumen:
```bash
python -m harris_corner.main --help
```

---

### 4. Menjalankan Unit Test
Untuk memastikan semua fungsi berjalan dengan benar:
```bash
pytest -v
```

---

## Output

Hasil proses akan tersimpan di folder `outputs/`:
* `grayscale.png`: Citra input dalam skala abu-abu.
* `corner_score.png`: Heatmap SSD Corner Score $E(u, v)$.
* `harris_response.png`: Heatmap nilai respons Harris $R$.
* `harris_corners.png`: Gambar asli dengan penanda titik sudut merah.
* `summary.png`: Ringkasan 4-panel (Original, Grayscale, SSD, Corners).
* `corner_score.npy` & `harris_response.npy`: Data mentah matriks numerik NumPy.
