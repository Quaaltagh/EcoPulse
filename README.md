# EcoPulse — Dashboard Klasifikasi Keberlanjutan Energi Negara

Web app Streamlit untuk men-deploy hasil clustering K-Means dari notebook
`DBSCAN_revisi_ML_final.ipynb` (176 negara, 5 indikator energi, k=2).

## Isi folder

| File | Keterangan |
|---|---|
| `app.py` | Aplikasi Streamlit utama (4 halaman, lihat di bawah) |
| `transformers.py` | Fungsi `clip_transform` & `log_transform` — **wajib ada**, dipakai pipeline preprocessing |
| `artifacts.pkl` | Model & metadata yang sudah dilatih (KMeans, pipeline, profil cluster, dll) — siap pakai |
| `build_artifacts.py` | Script untuk regenerasi `artifacts.pkl` dari ulang (lihat bagian "Regenerasi Model") |
| `requirements.txt` | Daftar dependency Python |

Empat halaman di `app.py`:
1. **Ringkasan Proyek** — overview dataset, metrik model, profil 2 cluster
2. **Prediksi Negara Baru** — input 5 indikator lewat slider, langsung dapat cluster + perbandingan
3. **Eksplorasi Cluster** — profil rata-rata per cluster, peta t-SNE interaktif, daftar negara per cluster
4. **Perbandingan Model** — tabel KMeans vs Hierarchical vs DBSCAN beserta alasan pemilihan model final

## Menjalankan di lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

Buka `http://localhost:8501` di browser.

## Deploy ke Streamlit Community Cloud (gratis)

1. Push folder ini (`app.py`, `transformers.py`, `artifacts.pkl`, `requirements.txt`) ke repo GitHub baru.
   File `.csv` dataset **tidak perlu** diikutkan — `artifacts.pkl` sudah memuat model yang jadi.
2. Buka [share.streamlit.io](https://share.streamlit.io) → **New app** → pilih repo tadi.
3. Isi *Main file path* dengan `app.py`, lalu **Deploy**.
4. Tunggu build selesai (1-3 menit) — app langsung online dengan URL publik.

## Regenerasi Model (opsional)

`artifacts.pkl` yang disertakan sudah diverifikasi cocok 100% dengan output notebook
(Silhouette KMeans 0.4657, DBI 0.9353, CHI 133.9227 — sama persis dengan hasil di
`DBSCAN_revisi_ML_final.ipynb`). Biasanya **tidak perlu** diregenerasi.

Regenerasi hanya disarankan kalau muncul error versi scikit-learn saat `joblib.load`
(pesan seperti *"incompatible version"*) di environment deploy kamu. Caranya:

```bash
kaggle datasets download -d anshtanwar/global-data-on-sustainable-energy
unzip global-data-on-sustainable-energy.zip
python build_artifacts.py   # akan menghasilkan artifacts.pkl baru
```

Versi library yang dipakai saat membuat `artifacts.pkl` ini: scikit-learn 1.8.0,
pandas 3.0.2, numpy 2.4.4, joblib 1.5.3. Kalau environment deploy kamu beda jauh
dari versi ini, regenerasi di atas akan menyamakannya.

## Catatan

`Renewables (% equivalent primary energy)` dan beberapa kolom lain tidak dipakai —
notebook ini memilih 5 fitur final (lihat bagian "Feature Engineering" di notebook).
DBSCAN tidak dipakai sebagai model prediksi (dipakai hanya untuk deteksi anomali),
jadi halaman "Prediksi Negara Baru" murni memakai KMeans.
