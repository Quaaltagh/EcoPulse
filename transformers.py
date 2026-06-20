"""
Fungsi transformasi kustom untuk preprocessing pipeline.
Dipisah ke modul sendiri supaya bisa di-pickle/unpickle dengan benar
oleh joblib lintas proses (dipakai oleh build_artifacts.py saat training
dan oleh app.py saat load model untuk prediksi).
"""
import numpy as np


def clip_transform(X, bounds=None):
    """Clip tiap kolom fitur ke batas IQR yang sudah dihitung saat training."""
    X = X.copy()
    for col, (lo, hi) in bounds.items():
        X[col] = X[col].clip(lo, hi)
    return X


def log_transform(X, cols=None):
    """Terapkan log1p ke kolom-kolom yang skewness-nya tinggi saat training."""
    X = X.copy()
    for col in cols:
        X[col] = np.log1p(X[col])
    return X
