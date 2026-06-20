"""
EcoPulse - Klasifikasi Keberlanjutan Energi Negara
Streamlit app untuk deployment hasil clustering (K-Means) dari notebook
DBSCAN_revisi_ML_final.ipynb.

Jalankan lokal:  streamlit run app.py
"""
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from transformers import clip_transform, log_transform  # wajib di-import sebelum load pickle

st.set_page_config(
    page_title="EcoPulse - Klasifikasi Energi Negara",
    page_icon="🌍",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    return joblib.load("artifacts.pkl")


art = load_artifacts()
FEATURES = art["features"]
KMEANS = art["kmeans"]
PIPELINE = art["preprocessing_pipeline"]
CLUSTER_NAMES = art["cluster_names"]
DF = art["df_result_feat"]
COMPARISON = art["comparison"]
X_TSNE = art["X_tsne"]
BEST_K = art["best_k"]
NOISE_COUNTRIES = set(art["noise_countries"])

PROFILE = DF.groupby("cluster_kmeans")[FEATURES].mean()
FEATURE_RANGES = {
    f: (float(DF[f].min()), float(DF[f].max()), float(DF[f].median()))
    for f in FEATURES
}
CLUSTER_COLORS = {0: "#1D9E75", 1: "#D8743A"}

FEATURE_LABELS = {
    "Access to electricity (% of population)": "Akses Listrik (% populasi)",
    "Renewable energy share in the total final energy consumption (%)": "Porsi Energi Terbarukan (%)",
    "Low-carbon electricity (% electricity)": "Listrik Rendah Karbon (%)",
    "Primary energy consumption per capita (kWh/person)": "Konsumsi Energi per Kapita (kWh/orang)",
    "Energy intensity level of primary energy (MJ/$2017 PPP GDP)": "Intensitas Energi (MJ/$ PPP GDP)",
}

# ---------------------------------------------------------------- sidebar --
st.sidebar.title("🌍 EcoPulse")
st.sidebar.caption("Klasifikasi keberlanjutan energi 176 negara")
page = st.sidebar.radio(
    "Navigasi",
    ["Ringkasan Proyek", "Prediksi Negara Baru", "Eksplorasi Cluster", "Perbandingan Model"],
)
st.sidebar.divider()
st.sidebar.markdown(
    "**Model final:** K-Means (k=2)\n\n"
    "**Sumber data:** [Global Data on Sustainable Energy](https://www.kaggle.com/datasets/anshtanwar/global-data-on-sustainable-energy) (Kaggle)"
)

# ============================================================== PAGE 1 ====
if page == "Ringkasan Proyek":
    st.title("🌍 Klasifikasi Keberlanjutan Energi Negara")
    st.markdown(
        "Dashboard ini menampilkan hasil clustering **176 negara** berdasarkan "
        "5 indikator keberlanjutan energi, menggunakan model **K-Means** yang sudah "
        "dilatih di notebook (`DBSCAN_revisi_ML_final.ipynb`)."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah Negara", len(DF))
    c2.metric("Jumlah Cluster", BEST_K)
    c3.metric("Silhouette Score", f"{COMPARISON.loc[COMPARISON['Model']=='K-Means','Silhouette'].iloc[0]:.4f}")
    c4.metric("Fitur Dipakai", len(FEATURES))

    st.divider()
    st.subheader("Dua Profil Negara yang Terbentuk")
    cols = st.columns(2)
    for c, col in zip(sorted(CLUSTER_NAMES), cols):
        with col:
            st.markdown(f"#### Cluster {c}")
            st.markdown(f"**{CLUSTER_NAMES[c]}**")
            n_country = (DF["cluster_kmeans"] == c).sum()
            st.caption(f"{n_country} negara")
            for f in FEATURES:
                st.write(f"- {FEATURE_LABELS[f]}: **{PROFILE.loc[c, f]:.2f}**")

    st.divider()
    st.subheader("Lima Fitur yang Digunakan")
    st.markdown(
        "\n".join(f"- **{FEATURE_LABELS[f]}** — `{f}`" for f in FEATURES)
    )

    st.divider()
    st.subheader("Insight Utama")
    st.info(
        "Cluster dengan porsi energi terbarukan tinggi *bukan* selalu berarti transisi "
        "energi yang aktif — pada Cluster 1, angka ini tinggi justru karena konsumsi energi "
        "totalnya sangat rendah / didominasi biomassa tradisional, bukan karena adopsi "
        "energi bersih modern seperti pada negara maju."
    )

# ============================================================== PAGE 2 ====
elif page == "Prediksi Negara Baru":
    st.title("🔮 Prediksi Cluster untuk Negara / Skenario Baru")
    st.markdown(
        "Masukkan nilai 5 indikator di bawah ini (misalnya untuk negara yang belum ada "
        "di dataset, atau skenario kebijakan energi hipotetis), lalu lihat negara itu "
        "masuk ke cluster mana."
    )

    with st.form("predict_form"):
        cols = st.columns(2)
        input_vals = {}
        for i, f in enumerate(FEATURES):
            lo, hi, med = FEATURE_RANGES[f]
            with cols[i % 2]:
                input_vals[f] = st.slider(
                    FEATURE_LABELS[f],
                    min_value=float(np.floor(lo)),
                    max_value=float(np.ceil(hi)),
                    value=float(med),
                    help=f"Kolom asli: {f}",
                )
        submitted = st.form_submit_button("Prediksi Cluster", type="primary")

    if submitted:
        X_new = pd.DataFrame([input_vals])[FEATURES]
        X_new_scaled = PIPELINE.transform(X_new)
        cluster_id = int(KMEANS.predict(X_new_scaled)[0])
        distances = KMEANS.transform(X_new_scaled)[0]

        st.success(f"**Cluster {cluster_id}** — {CLUSTER_NAMES[cluster_id]}")

        c1, c2 = st.columns([1, 1.4])
        with c1:
            st.markdown("**Jarak ke setiap centroid** (makin kecil = makin mirip)")
            dist_df = pd.DataFrame({
                "Cluster": [f"Cluster {i} ({CLUSTER_NAMES[i]})" for i in range(len(distances))],
                "Jarak": distances.round(4),
            })
            st.dataframe(dist_df, hide_index=True, use_container_width=True)

        with c2:
            st.markdown("**Input vs rata-rata profil cluster terpilih**")
            compare_df = pd.DataFrame({
                "Fitur": [FEATURE_LABELS[f] for f in FEATURES],
                "Input Anda": [input_vals[f] for f in FEATURES],
                "Rata-rata Cluster": [PROFILE.loc[cluster_id, f] for f in FEATURES],
            }).melt(id_vars="Fitur", var_name="Sumber", value_name="Nilai")
            fig = px.bar(
                compare_df, x="Fitur", y="Nilai", color="Sumber", barmode="group",
                color_discrete_sequence=["#3B82C4", "#B4B2A9"],
            )
            fig.update_layout(xaxis_tickangle=-25, legend_title="", height=380)
            st.plotly_chart(fig, use_container_width=True)

# ============================================================== PAGE 3 ====
elif page == "Eksplorasi Cluster":
    st.title("📊 Eksplorasi Hasil Clustering")

    tab1, tab2, tab3 = st.tabs(["Profil Cluster", "Peta t-SNE", "Daftar Negara"])

    with tab1:
        st.subheader("Rata-rata Fitur per Cluster")
        display_profile = PROFILE.rename(columns=FEATURE_LABELS).round(2)
        display_profile.index = [f"Cluster {i} — {CLUSTER_NAMES[i]}" for i in display_profile.index]
        st.dataframe(display_profile.T, use_container_width=True)

        long_df = PROFILE.reset_index().melt(id_vars="cluster_kmeans", var_name="Fitur", value_name="Nilai")
        long_df["Fitur"] = long_df["Fitur"].map(FEATURE_LABELS)
        long_df["Cluster"] = long_df["cluster_kmeans"].map(lambda c: f"Cluster {c}")
        fig = px.bar(
            long_df, x="Fitur", y="Nilai", color="Cluster", barmode="group",
            color_discrete_map={f"Cluster {k}": v for k, v in CLUSTER_COLORS.items()},
        )
        fig.update_layout(xaxis_tickangle=-25, height=420)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Visualisasi 2D (t-SNE) Seluruh Negara")
        plot_df = DF.copy()
        plot_df["x"] = X_TSNE[:, 0]
        plot_df["y"] = X_TSNE[:, 1]
        plot_df["Cluster"] = plot_df["cluster_kmeans"].map(lambda c: f"Cluster {c}")
        plot_df["Status"] = plot_df["Entity"].apply(
            lambda e: "Anomali (DBSCAN noise)" if e in NOISE_COUNTRIES else "Normal"
        )
        fig = px.scatter(
            plot_df, x="x", y="y", color="Cluster", symbol="Status",
            hover_name="Entity",
            hover_data={f: True for f in FEATURES} | {"x": False, "y": False},
            color_discrete_map={f"Cluster {k}": v for k, v in CLUSTER_COLORS.items()},
        )
        fig.update_layout(height=550, xaxis_title="Komponen t-SNE 1", yaxis_title="Komponen t-SNE 2")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Simbol berbeda menandai negara yang oleh DBSCAN dianggap anomali/outlier "
            "energi (profil terlalu unik untuk masuk cluster manapun)."
        )

    with tab3:
        st.subheader("Daftar Negara per Cluster")
        c_filter = st.selectbox(
            "Pilih cluster",
            options=sorted(CLUSTER_NAMES),
            format_func=lambda c: f"Cluster {c} — {CLUSTER_NAMES[c]}",
        )
        search = st.text_input("Cari negara", "")
        subset = DF[DF["cluster_kmeans"] == c_filter][["Entity"] + FEATURES].rename(columns=FEATURE_LABELS)
        if search:
            subset = subset[subset["Entity"].str.contains(search, case=False)]
        st.dataframe(subset.reset_index(drop=True), use_container_width=True, height=420)
        st.caption(f"{len(subset)} negara ditampilkan.")

# ============================================================== PAGE 4 ====
elif page == "Perbandingan Model":
    st.title("⚖️ Perbandingan Algoritma Clustering")
    st.markdown(
        "Tiga algoritma dibandingkan dengan k yang sama (k=2 untuk K-Means & Hierarchical). "
        "DBSCAN dievaluasi terpisah karena sebagian besar titik dianggap *noise*."
    )

    def highlight(row):
        styles = [""] * len(row)
        return styles

    styled = COMPARISON.style.highlight_max(subset=["Silhouette", "CHI"], color="#1D9E75") \
                              .highlight_min(subset=["DBI"], color="#1D9E75")
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown(
        "**Kriteria:** Silhouette & CHI semakin tinggi semakin baik; DBI semakin rendah semakin baik."
    )

    st.divider()
    st.subheader("Mengapa K-Means Dipilih sebagai Model Final?")
    st.markdown(
        "- Menggunakan **seluruh** data negara dalam proses clustering, hasil lebih representatif.\n"
        f"- DBSCAN menganggap **{COMPARISON.loc[COMPARISON['Model']=='DBSCAN*','Noise (%)'].iloc[0]}%** "
        "data sebagai noise — sebagian besar negara tidak masuk cluster manapun.\n"
        "- DBI K-Means lebih rendah dan CHI lebih tinggi dibanding Hierarchical Clustering.\n"
        "- Cluster yang terbentuk lebih mudah diinterpretasikan sesuai tujuan riset."
    )

    with st.expander("Lihat negara yang dianggap anomali oleh DBSCAN"):
        st.write(f"{len(NOISE_COUNTRIES)} negara dengan profil energi terlalu unik untuk dikelompokkan:")
        st.write(", ".join(sorted(NOISE_COUNTRIES)))
