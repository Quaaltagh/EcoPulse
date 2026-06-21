# 🌍 EcoPulse — Klasifikasi Keberlanjutan Energi Negara

Aplikasi web (Streamlit) untuk mengelompokkan 176 negara berdasarkan indikator
keberlanjutan energi, menggunakan model **K-Means** yang dilatih di
`DBSCAN_revisi_ML_final.ipynb`. Proyek ML — BINUS Semester 4.

## Ringkasan

Setiap negara dikelompokkan ke salah satu dari 2 cluster berdasarkan 5 indikator
energi (rata-rata 3 tahun terakhir per negara). Tujuannya membedakan negara
dengan **akses & konsumsi energi modern yang tinggi** dari negara dengan
**akses terbatas, konsumsi rendah yang sebagian besar masih berbasis biomassa
tradisional** — bukan transisi energi bersih yang sesungguhnya.

| | |
|---|---|
| **Dataset** | [Global Data on Sustainable Energy](https://www.kaggle.com/datasets/anshtanwar/global-data-on-sustainable-energy) (Kaggle), 176 negara, 2000–2020 |
| **Fitur** | 5 indikator (lihat tabel di bawah) |
| **Model final** | K-Means, k=2 (dipilih dari pencarian k=2–10 berdasarkan Silhouette Score tertinggi) |
| **Pembanding** | Hierarchical Clustering (Ward linkage), DBSCAN (deteksi anomali) |

## Struktur Proyek

```
ECOPLUS/
├── app.py                          # Aplikasi Streamlit (3 halaman, lihat di bawah)
├── requirements.txt
├── model/
│   ├── kmeans_model.pkl            # Model K-Means terlatih (k=2)
│   ├── clip_bounds.pkl             # Batas IQR clipping per fitur (dict: kolom -> (lower, upper))
│   ├── log_transformed_features.pkl# Daftar fitur yang di-log1p (skewness tinggi)
│   ├── features.pkl                # Urutan 5 nama kolom fitur final
│   ├── cluster_profiles.pkl        # DataFrame per negara: Entity, cluster_kmeans, nilai fitur (clipped)
│   ├── cluster_names.pkl           # Nama deskriptif tiap cluster {0: "...", 1: "..."}
│   ├── X_log.pkl                   # Data ter-clip+log (dipakai utk fit ulang RobustScaler saat app start)
│   ├── noise_countries.pkl         # Negara yang dianggap anomali oleh DBSCAN
│   └── preprocessing_pipeline.pkl  # (lihat catatan di bawah — saat ini tidak dipakai app.py)
```

App **tidak** memuat `preprocessing_pipeline.pkl` langsung. `app.py` membangun
ulang pipeline (`clip → log1p → RobustScaler`) di fungsi `_build_pipeline()` dari
`clip_bounds.pkl` + `log_transformed_features.pkl` + `X_log.pkl`, lalu fit
ulang scaler-nya saat startup. Ini sengaja, supaya tidak tergantung pickle
`Pipeline`/`FunctionTransformer` yang rawan gagal saat versi scikit-learn di
Colab (tempat training) beda dengan versi di environment deploy.

Tiga halaman di `app.py`:
1. **Prediksi Negara Baru** — input 5 indikator, dapat cluster + jarak ke centroid + radar chart
2. **Eksplorasi Cluster** — profil tiap cluster, peta t-SNE interaktif, daftar negara per cluster
3. **Tentang Model** — metodologi, dataset, dan tabel perbandingan algoritma

## Metodologi Singkat

| Tahap | Detail |
|---|---|
| Agregasi | Rata-rata 3 tahun data terakhir per negara |
| Missing value | Median imputation per fitur |
| Outlier | IQR clipping (winsorizing) |
| Transformasi | log1p untuk fitur dengan skewness > 1.0 |
| Scaling | RobustScaler |
| Pemilihan k | Silhouette Score, scan k=2–10 |

## Hasil

| Model | Silhouette ↑ | DBI ↓ | CHI ↑ | Noise |
|---|---|---|---|---|
| **K-Means (final)** | **0.4657** | **0.9353** | **133.92** | 0% |
| Hierarchical (Ward) | 0.4432 | 0.9821 | 118.45 | 0% |
| DBSCAN* | 0.5417 | 0.5886 | 26.74 | 92.6% |

\*DBSCAN dievaluasi hanya pada titik non-noise — tidak dipakai sebagai model
prediksi karena >90% negara dianggap noise, hanya untuk deteksi anomali.

**Cluster 0** — akses listrik & konsumsi energi tinggi, efisiensi lebih baik.
**Cluster 1** — akses & konsumsi energi rendah, energi terbarukan didominasi
biomassa tradisional (bukan solar/wind).

## Cara Menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

Buka `http://localhost:8501`. Folder `model/` harus ada di direktori yang sama
dengan `app.py` (path di kode memakai `./model/...`).

## Deploy ke Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, dan folder `model/` ke repo GitHub.
2. Buka [share.streamlit.io](https://share.streamlit.io) → **New app** → pilih repo, isi main file `app.py` → **Deploy**.

## Catatan Maintenance

`PROFILE_AVG` dan deskripsi naratif tiap cluster di `CLUSTER_CONFIG` saat ini
**di-hardcode di `app.py`**, tidak dihitung otomatis dari `cluster_profiles.pkl`.
Kalau model di-retrain (fitur berubah, k berubah, dataset di-update), dua hal
itu **tidak ikut berubah otomatis** — perlu diperbarui manual supaya tidak
menampilkan profil yang sudah tidak sesuai dengan model terbaru.

Kalau muncul `InconsistentVersionWarning` dari scikit-learn saat `joblib.load`,
itu tanda versi scikit-learn saat training (Colab) beda dengan versi lokal —
biasanya cuma warning, tapi kalau sampai error, samakan versi scikit-learn
di `requirements.txt` dengan yang dipakai saat training di notebook.

## Sumber Data

Ansh Tanwar, [Global Data on Sustainable Energy](https://www.kaggle.com/datasets/anshtanwar/global-data-on-sustainable-energy), Kaggle.