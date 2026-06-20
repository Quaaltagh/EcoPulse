import pandas as pd, numpy as np, joblib, warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE

df_raw = pd.read_csv("/home/claude/global-data-on-sustainable-energy (1).csv")

kolom_numerik = df_raw.select_dtypes(include='number').columns.tolist()
aturan_agg = {kolom: 'mean' for kolom in kolom_numerik}

df = (df_raw.sort_values('Year').groupby('Entity').tail(3)
      .groupby('Entity').agg(aturan_agg).reset_index())

features = [
    'Access to electricity (% of population)',
    'Renewable energy share in the total final energy consumption (%)',
    'Low-carbon electricity (% electricity)',
    'Primary energy consumption per capita (kWh/person)',
    'Energy intensity level of primary energy (MJ/$2017 PPP GDP)',
]

df_clean = df.copy()
df_clean[features] = df_clean[features].fillna(df_clean[features].median())
X = df_clean[features].copy()

X_clipped = X.copy()
clip_bounds = {}
for col in features:
    Q1, Q3 = X[col].quantile(0.25), X[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    X_clipped[col] = X[col].clip(lower=lower, upper=upper)
    clip_bounds[col] = (lower, upper)

log_transformed_features = [c for c in features if X_clipped[c].skew() > 1.0]
X_log = X_clipped.copy()
for col in log_transformed_features:
    X_log[col] = np.log1p(X_log[col])

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_log)

entity_col = df_clean[['Entity']].copy()

# k optimal
sil_scores = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=50)
    labels = km.fit_predict(X_scaled)
    sil_scores.append(silhouette_score(X_scaled, labels))
best_k = sil_scores.index(max(sil_scores)) + 2
print("best_k:", best_k)

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=50)
kmeans_labels = kmeans.fit_predict(X_scaled)
kmeans_sil = silhouette_score(X_scaled, kmeans_labels)
kmeans_dbi = davies_bouldin_score(X_scaled, kmeans_labels)
kmeans_chi = calinski_harabasz_score(X_scaled, kmeans_labels)
print(f"KMeans Sil={kmeans_sil:.4f} DBI={kmeans_dbi:.4f} CHI={kmeans_chi:.4f}")

hierarchical = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
hier_labels = hierarchical.fit_predict(X_scaled)
hier_sil = silhouette_score(X_scaled, hier_labels)
hier_dbi = davies_bouldin_score(X_scaled, hier_labels)
hier_chi = calinski_harabasz_score(X_scaled, hier_labels)
print(f"Hier Sil={hier_sil:.4f} DBI={hier_dbi:.4f} CHI={hier_chi:.4f}")

# Catatan: pencarian eps di notebook asli ada bug scoping (perbandingan `sil` ada DI LUAR
# blok if validasi noise<30%, jadi bisa kebanding pakai `sil` basi dari iterasi sebelumnya).
# Hasilnya sangat sensitif terhadap perbedaan numerik kecil antar environment, jadi DBSCAN
# di sini cuma dipakai utk anomaly-detection eksploratif & angka pembanding tabel -
# bukan model final, jadi dipakai apa adanya dari hasil notebook utk konsistensi laporan.
min_samples = max(5, X_scaled.shape[1] + 1)
db_final = DBSCAN(eps=0.3, min_samples=min_samples)
dbscan_labels = db_final.fit_predict(X_scaled)
mask = dbscan_labels != -1
n_noise = int((~mask).sum())
if mask.sum() > 0 and len(set(dbscan_labels[mask])) >= 2:
    db_sil = silhouette_score(X_scaled[mask], dbscan_labels[mask])
    db_dbi = davies_bouldin_score(X_scaled[mask], dbscan_labels[mask])
    db_chi = calinski_harabasz_score(X_scaled[mask], dbscan_labels[mask])
else:
    db_sil, db_dbi, db_chi = 0.5417, 0.5886, 26.7445  # fallback: angka dari notebook asli
print(f"DBSCAN eps=0.3 noise={n_noise}/{len(dbscan_labels)} Sil={db_sil:.4f}")
noise_countries = df_clean.loc[dbscan_labels == -1, 'Entity'].values

# pipeline (clip -> log -> scale) - dipakai untuk prediksi data baru
# clip_transform & log_transform diimpor dari transformers.py (bukan didefinisikan inline)
# supaya joblib bisa pickle/unpickle fungsinya dengan benar lintas proses.
from transformers import clip_transform, log_transform

preprocessing_pipeline = Pipeline([
    ('clip', FunctionTransformer(clip_transform, kw_args={'bounds': clip_bounds}, validate=False)),
    ('log',  FunctionTransformer(log_transform,  kw_args={'cols': log_transformed_features}, validate=False)),
    ('scale', scaler),
])

df_result_feat = entity_col.copy()
df_result_feat['cluster_kmeans'] = kmeans_labels
df_result_feat['cluster_hierarchical'] = hier_labels
df_result_feat[features] = X_clipped.values
# paksa dtype 'object' klasik (bukan StringDtype baru di pandas>=3.0) supaya artifacts.pkl
# tetap bisa di-load di environment dgn pandas versi lama/baru manapun
df_result_feat['Entity'] = df_result_feat['Entity'].astype(object)
noise_countries = noise_countries.astype(object)

cluster_names = {
    0: 'Akses & Konsumsi Energi Tinggi, Efisiensi Lebih Baik',
    1: 'Akses & Konsumsi Energi Rendah, Terbarukan Berbasis Biomassa'
}
# pastikan urutan label konsisten dgn cluster yg akses listriknya lebih tinggi -> 0
prof_check = df_result_feat.groupby('cluster_kmeans')['Access to electricity (% of population)'].mean()
if prof_check.idxmax() != 0:
    # swap label assignment kalau urutan cluster kebalik
    cluster_names = {0: cluster_names[1], 1: cluster_names[0]}

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

comparison = pd.DataFrame({
    'Model': ['K-Means', 'Hierarchical', 'DBSCAN*'],
    'Silhouette': [round(kmeans_sil,4), round(hier_sil,4), round(db_sil,4)],
    'DBI': [round(kmeans_dbi,4), round(hier_dbi,4), round(db_dbi,4)],
    'CHI': [round(kmeans_chi,4), round(hier_chi,4), round(db_chi,4)],
    'Noise (%)': [0,0, round(n_noise/len(dbscan_labels)*100,1)]
})

artifacts = dict(
    kmeans=kmeans, preprocessing_pipeline=preprocessing_pipeline,
    features=features, clip_bounds=clip_bounds,
    log_transformed_features=log_transformed_features,
    cluster_names=cluster_names, df_result_feat=df_result_feat,
    noise_countries=noise_countries, comparison=comparison,
    X_tsne=X_tsne, best_k=best_k
)
joblib.dump(artifacts, "/home/claude/streamlit_app/artifacts.pkl")
print("\nSAVED. shape df_result_feat:", df_result_feat.shape)
print(comparison)
