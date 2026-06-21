import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.manifold import TSNE

st.markdown("""
	<style>
	.cluster-banner {
		border-radius: 16px;
		padding: 28px 32px;
		border: 1px solid;
	}
	
	.cluster-0 {
		background: rgba(37, 99, 235, 0.1);

		border-color: rgba(37, 99, 235, 0.5);
	}
		
	.cluster-1 {
		background: rgba(37, 235, 73, 0.1);

		border-color: rgba(37, 235, 73, 0.5);
	}
		
	.cluster-banner .cluster-num {
		font-size: 12px;
		font-weight: 800;
		letter-spacing: 0.1em;
		text-transform: uppercase;
		margin-bottom: 6px;
	}
		
	.cluster-0 .cluster-num { color: #2563EB;}
	.cluster-1 .cluster-num { color: #4ADE80; }
				
	.cluster-banner .cluster-name {
		font-size: 22px;
		font-weight: 700;
		line-height: 1.3;
		margin: 0 0 10px 0;
	}
				
	.cluster-banner .cluster-desc {
		font-size: 14px;
		line-height: 1.6;
	}
		
	.metric-card {
		background: rgba(100, 118, 143, 0.1);
		border: 1px solid rgba(100, 118, 143, 0.5);
		border-radius: 12px;
		padding: 20px 24px;
		margin-bottom: 12px;
	}
		
	.metric-card h3 {
		font-size: 16px;
		font-weight: 500;
		text-transform: uppercase;
		letter-spacing: 0.08em;
		padding: 0.75rem 0px .5rem;
	}
	.metric-card .val {
		font-size: 28px;
		font-weight: 700;
		line-height: 1;
	}
		
	.metric-card .sub {
		font-size: 14px;
		margin-top: 4px;
	}
			
	/* Section title */
	.section-title {
		font-size: 1.75rem;
    font-weight: 600;
    padding: 0.75rem 0px 1rem;
		margin-top:0.75rem;
	}
			
	.country-tag-0 {
		display: inline-block;
		background: rgba(37, 99, 235, 0.1);
		border-radius: 6px;
		padding: 3px 10px;
		font-size: 12px;
		margin: 3px;
	}
			
	.country-tag-1 {
		display: inline-block;
		background: rgba(37, 235, 73, 0.1);
		border-radius: 6px;
		padding: 3px 10px;
		font-size: 12px;
		margin: 3px;
	}
	
	.country-tag-2 {
		display: inline-block;
		background: rgba(112, 112, 112, 0.1);
		border-radius: 6px;
		padding: 3px 10px;
		font-size: 12px;
		margin: 3px;
	}
			
	/* Insight box */
	.insight-box {
		background-color: rgba(211, 211, 211, 0.2);
		border-left: 3px solid #3B82F6;
		border-radius: 0 8px 8px 0;
		padding: 12px 16px;
		margin: 10px 0;
	}
	.insight-box.green { border-color: #22C55E; }
	.insight-box.amber { border-color: #F59E0B; }
	.insight-box.orange { border-color: rgb(245, 158, 11); }
	.insight-box p { margin: 0; line-height: 1.6; }
	
	</style>
	""", unsafe_allow_html=True)



st.set_page_config(
    page_title="Energy Sustainability Clustering",
    layout="wide",
    initial_sidebar_state="expanded",
)

def _build_pipeline(clip_bounds, log_features, X_train):
	from sklearn.pipeline import Pipeline
	from sklearn.preprocessing import FunctionTransformer, RobustScaler

	_cb = clip_bounds
	_lf = log_features

	def _clip(X):
			X = X.copy()
			for col, (lo, hi) in _cb.items():
					if col in X.columns:
							X[col] = X[col].clip(lo, hi)
			return X

	def _log(X):
			X = X.copy()
			for col in _lf:
					if col in X.columns:
							X[col] = np.log1p(X[col])
			return X

	pipe = Pipeline([
			("clip",  FunctionTransformer(_clip, validate=False)),
			("log",   FunctionTransformer(_log,  validate=False)),
			("scale", RobustScaler()),
	])
	pipe.fit(X_train)
	return pipe


@st.cache_resource
def load_artifacts():
	try:
		kmeans= joblib.load("./model/kmeans_model.pkl")
		clip_bounds = joblib.load("./model/clip_bounds.pkl")
		log_feats= joblib.load("./model/log_transformed_features.pkl")
		features= joblib.load("./model/features.pkl")
		profiles= joblib.load("./model/cluster_profiles.pkl")
		c_names = joblib.load("./model/cluster_names.pkl")
		X_train = joblib.load("./model/X_log.pkl")
		noise_countries = joblib.load("./model/noise_countries.pkl")
		pipeline = _build_pipeline(clip_bounds, log_feats, X_train)

		return kmeans, pipeline, features, profiles, c_names, None, noise_countries
	except Exception as e:
			return None, None, None, None, None, str(e), None

kmeans, pipeline, features, df_profiles, cluster_names, load_error, noise_countries = load_artifacts()
if load_error:
	st.error(f"Gagal load model dari folder ./model/ : {load_error}")
	st.stop()


@st.cache_data
def compute_tsne_coords(_pipeline, feature_df):
	"""Hitung koordinat t-SNE 2D untuk seluruh negara (sekali saja, lalu di-cache)."""
	X_scaled_all = _pipeline.transform(feature_df)
	return TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_scaled_all)


FEATURE_CONFIG = {
    "Access to electricity (% of population)": {
        "label": "Akses Listrik",
        "unit": "%",
        "min": 0.0, "max": 100.0, "default": 80.0, "step": 0.5,
        "help": "Persentase populasi dengan akses listrik",
    },
    "Renewable energy share in the total final energy consumption (%)": {
        "label": "Porsi Energi Terbarukan",
        "unit": "%",
        "min": 0.0, "max": 100.0, "default": 30.0, "step": 0.5,
        "help": "Persentase energi terbarukan dari total konsumsi energi final",
    },
    "Low-carbon electricity (% electricity)": {
        "label": "Listrik Rendah Karbon",
        "unit": "%",
        "min": 0.0, "max": 100.0, "default": 35.0, "step": 0.5,
        "help": "Persentase listrik dari sumber rendah karbon (terbarukan + nuklir)",
    },
    "Primary energy consumption per capita (kWh/person)": {
        "label": "Konsumsi Energi per Kapita",
        "unit": "kWh/orang",
        "min": 0.0, "max": 80000.0, "default": 10000.0, "step": 100.0,
        "help": "Total konsumsi energi primer per orang per tahun",
    },
    "Energy intensity level of primary energy (MJ/$2017 PPP GDP)": {
        "label": "Intensitas Energi",
        "unit": "MJ/$GDP",
        "min": 0.0, "max": 20.0, "default": 5.0, "step": 0.1,
        "help": "Energi yang dibutuhkan per unit GDP — semakin rendah semakin efisien",
    },
}

CLUSTER_CONFIG = {
    0: {
        "color": "#2563EB",
        "light": "#60A5FA",
        "bg": "cluster-0",
        "emoji": "⚡",
        "desc": "Negara dengan infrastruktur energi modern dan akses listrik hampir universal, namun masih bergantung besar pada bahan bakar fosil. Energy intensity rendah menunjukkan efisiensi ekonomi yang baik.",
        "examples": ["Amerika Serikat", "Jerman", "Jepang", "Korea Selatan", "Australia", "Prancis"],
    },
    1: {
        "color": "#16A34A",
        "light": "#4ADE80",
        "bg": "cluster-1",
        "emoji": "🌿",
        "desc": "Negara dengan akses listrik terbatas dan konsumsi energi sangat rendah. Renewable share tinggi didominasi biomassa tradisional (kayu bakar, arang) — bukan solar/wind. Konsumsi rendah bukan karena efisien, tapi karena keterbatasan akses.",
        "examples": ["Ethiopia", "Bangladesh", "Niger", "Chad", "Uganda", "Mozambik"],
    },
}

PROFILE_AVG = {
    0: {
        "Access to electricity (% of population)": 96.72,
        "Renewable energy share in the total final energy consumption (%)": 19.98,
        "Low-carbon electricity (% electricity)": 36.35,
        "Primary energy consumption per capita (kWh/person)": 27857.69,
        "Energy intensity level of primary energy (MJ/$2017 PPP GDP)": 3.91,
    },
    1: {
        "Access to electricity (% of population)": 49.94,
        "Renewable energy share in the total final energy consumption (%)": 68.59,
        "Low-carbon electricity (% electricity)": 48.36,
        "Primary energy consumption per capita (kWh/person)": 1976.34,
        "Energy intensity level of primary energy (MJ/$2017 PPP GDP)": 5.77,
    },
}

# side bar
st.sidebar.title("🌍 EcoPulse")
st.sidebar.caption("Klasifikasi keberlanjutan energi 176 negara")
page = st.sidebar.radio(
    "Navigasi",
    ["Prediksi Negara Baru", "Eksplorasi Cluster", "Tentang Model"],
)
st.sidebar.divider()
st.sidebar.markdown(
    "**Model final:** K-Means (k=2)\n\n"
		"**Silhouette:** 0.4657\n\n"
		"**DBI:** 0.9353\n\n"
		"**CHI :** 133.92\n\n"
    "**Sumber data:** [Global Data on Sustainable Energy](https://www.kaggle.com/datasets/anshtanwar/global-data-on-sustainable-energy) (Kaggle)"
)

# prediksi

if "Prediksi" in page:
	st.title("Prediksi Cluster Negara")
	st.text("Masukkan indikator energi suatu negara untuk mengetahui kelompok sustainability-nya")

	col_form, col_result = st.columns([1, 1], gap="large")

	with col_form:
		st.title('Input Indikator')
		st.text('Semua nilai berdasarkan rata-rata 3 tahun terakhir negara tersebut')

		input_vals = {}
		for feat, cfg in FEATURE_CONFIG.items():
			input_vals[feat] = st.number_input(
				f"{cfg['label']} ({cfg['unit']})",
				min_value=cfg["min"],
				max_value=cfg["max"],
				value=cfg["default"],
				step=cfg["step"],
				help=cfg["help"],
				key=feat,
			)
		
		predict_btn = st.button("Prediksi Cluster", use_container_width=True)

	with col_result:
		st.title('Hasil Prediksi')
		st.text('Cluster dan profil berdasarkan model K-Means')

		if predict_btn:
			X_new = pd.DataFrame([input_vals])[features]
			X_new_scaled = pipeline.transform(X_new)
			cluster_id = int(kmeans.predict(X_new_scaled)[0])
			distances = kmeans.transform(X_new_scaled)[0]
			cfg = CLUSTER_CONFIG[cluster_id]
			c_name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")

			st.markdown(f"""
				<div class='cluster-banner {cfg["bg"]}'>
					<div class='cluster-num'>{cfg["emoji"]} Cluster {cluster_id}</div>
					<div class='cluster-name'>{c_name}</div>
					<div class='cluster-desc'>{cfg["desc"]}</div>
				</div>
				""", unsafe_allow_html=True)
			
			total_dist = sum(distances)
			confidence = (1 - distances[cluster_id] / total_dist) * 100
			conf_color = "#22C55E" if confidence > 70 else "#F59E0B" if confidence > 50 else "#EF4444"

			st.markdown(f"""
			<div style='display:flex; gap:16px; margin:12px 0;'>
					<div class='metric-card' style='flex:1'>
							<h3>Keyakinan Model</h3>
							<div class='val' style='color:{conf_color}'>{confidence:.1f}%</div>
							<div class='sub'>Jarak ke centroid: {distances[cluster_id]:.4f}</div>
					</div>
					<div class='metric-card' style='flex:1'>
							<h3>Cluster Lainnya</h3>
							<div class='val' style='font-size:20px'>Cluster {1 - cluster_id}</div>
							<div class='sub'>Jarak: {distances[1 - cluster_id]:.4f}</div>
					</div>
			</div>
			""", unsafe_allow_html=True)

			avg = PROFILE_AVG[cluster_id]
			labels_radar = [FEATURE_CONFIG[f]["label"] for f in features]

			def normalize(vals_dict):
				result = []
				for f in features:
					lo = FEATURE_CONFIG[f]["min"]
					hi = FEATURE_CONFIG[f]["max"]
					result.append((vals_dict[f] - lo) / (hi - lo + 1e-9))
				return result
			
			fig_radar = go.Figure()
			fig_radar.add_trace(go.Scatterpolar(
					r=normalize(input_vals) + [normalize(input_vals)[0]],
					theta=labels_radar + [labels_radar[0]],
					fill="toself",
					name="Input Kamu",
					line_color="#3B82F6",
					fillcolor="rgba(59,130,246,0.15)",
			))
			fig_radar.add_trace(go.Scatterpolar(
					r=normalize(avg) + [normalize(avg)[0]],
					theta=labels_radar + [labels_radar[0]],
					fill="toself",
					name=f"Rata-rata Cluster {cluster_id}",
					line_color=cfg["color"],
					fillcolor=f"rgba({','.join(str(int(cfg['color'].lstrip('#')[i:i+2],16)) for i in (0,2,4))},0.12)",
			))
			fig_radar.update_layout(
					polar=dict(
							bgcolor="rgba(0,0,0,0)",
							radialaxis=dict(visible=True, range=[0, 1], ),
							angularaxis=dict(gridcolor="#1F2937", ),
					),
					paper_bgcolor="rgba(0,0,0,0)",
					plot_bgcolor="rgba(0,0,0,0)",
					showlegend=True,
					legend=dict(font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
					margin=dict(l=40, r=40, t=30, b=30),
					height=320,
			)
			st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})


		else:
			st.markdown(
					"""
					<div style=
					"text-align: center;
					display:flex;
					flex-direction:column;
					justify-content:center;
					align-items:center;
					height:250px;
					border:1px dashed #d1d5db;
					border-radius:10px;
					">
							<p><b>Isi indikator dan klik Prediksi</b></p>
							<p>Hasil cluster akan muncul di sini</p>
					</div>
					""",
					unsafe_allow_html=True
			)

elif "Eksplorasi" in page:
	st.title("📊 Eksplorasi Hasil Clustering")
	tab1, tab2, tab3 = st.tabs(["Profil Cluster", "Peta t-SNE", "Daftar Negara"])

	with tab1:
		st.subheader("Profil Cluster")

		col0, col1 = st.columns(2, gap="large")
		for idx, col in enumerate([col0, col1]):
				cfg = CLUSTER_CONFIG[idx]
				c_name = cluster_names.get(idx, f"Cluster {idx}") if cluster_names else f"Cluster {idx}"
				avg = PROFILE_AVG[idx]

				with col:
					st.markdown(f"""
					<div class='cluster-banner {cfg["bg"]}'>
							<div class='cluster-num'>{cfg["emoji"]} Cluster {idx}</div>
							<div class='cluster-name'>{c_name}</div>
							<div class='cluster-desc'>{cfg["desc"]}</div>
					</div>
					""", unsafe_allow_html=True)
					for feat, val in avg.items():
							label = FEATURE_CONFIG[feat]["label"]
							unit  = FEATURE_CONFIG[feat]["unit"]
							hi    = FEATURE_CONFIG[feat]["max"]
							pct   = min(val / hi, 1.0)
							color = cfg["color"]
							st.markdown(f"""
							<div style='padding:8px 0; border-bottom:1px solid '>
									<div style='display:flex; justify-content:space-between; margin-bottom:4px;'>
											<span style='font-size:13px;'>{label}</span>
											<span style='font-size:13px; font-weight:600;'>{val:,.2f} {unit}</span>
									</div>
									<div style='background:lightgray; height:4px; border-radius:2px;'>
											<div style='background:{color}; width:{pct*100:.1f}%; height:100%; border-radius:2px;'></div>
									</div>
							</div>
							""", unsafe_allow_html=True)
						
		# barchart
		st.markdown("<div class='section-title'>Perbandingan Fitur antar Cluster</div>", unsafe_allow_html=True)
		st.text('Nilai rata-rata dinormalisasi (0–1) untuk memudahkan perbandingan skala berbeda')

		feat_labels = [FEATURE_CONFIG[f]["label"] for f in features]
		norm0 = [min(PROFILE_AVG[0][f] / FEATURE_CONFIG[f]["max"], 1.0) for f in features]
		norm1 = [min(PROFILE_AVG[1][f] / FEATURE_CONFIG[f]["max"], 1.0) for f in features]

		fig_bar = go.Figure()
		fig_bar.add_trace(go.Bar(name="Cluster 0", x=feat_labels, y=norm0, marker_color="#2563EB", opacity=0.85))
		fig_bar.add_trace(go.Bar(name="Cluster 1", x=feat_labels, y=norm1, marker_color="#16A34A", opacity=0.85))
		fig_bar.update_layout(
				barmode="group",
				paper_bgcolor="rgba(0,0,0,0)",
				yaxis=dict(title="Nilai Normalisasi (0–1)"),
		)
		st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

	with tab2:
		st.subheader("Visualisasi 2D (t-SNE) Seluruh Negara")

		coords = compute_tsne_coords(pipeline, df_profiles[features])
		plot_df = df_profiles.copy()
		plot_df["x"] = coords[:, 0]
		plot_df["y"] = coords[:, 1]
		plot_df["Cluster"] = plot_df["cluster_kmeans"].map(lambda c: f"Cluster {c}")
		plot_df["Status"] = plot_df["Entity"].apply(
				lambda e: "Anomali (DBSCAN noise)" if e in noise_countries else "Normal"
		)
		cluster_color_map = {f"Cluster {k}": v["color"] for k, v in CLUSTER_CONFIG.items()}
		fig = px.scatter(
				plot_df, x="x", y="y", color="Cluster", symbol="Status",
				hover_name="Entity",
				hover_data={f: True for f in features} | {"x": False, "y": False},
				color_discrete_map=cluster_color_map,
		)
		fig.update_layout(
				height=550, xaxis_title="Komponen t-SNE 1", yaxis_title="Komponen t-SNE 2",
				paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
		)
		st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
		st.caption(
				"Simbol berbeda menandai negara yang oleh DBSCAN dianggap anomali/outlier "
				"energi (profil terlalu unik untuk masuk cluster manapun)."
		)
	
	with tab3:
		st.subheader("Daftar Negara per Cluster")
		tab0, tab1 = st.tabs([
				f"⚡ Cluster 0 — {cluster_names.get(0, 'Cluster 0')}",
				f"🌿 Cluster 1 — {cluster_names.get(1, 'Cluster 1')}"
		])

		for tab, cluster_id in [(tab0, 0), (tab1, 1)]:
				with tab:
						cfg = CLUSTER_CONFIG[cluster_id]
						countries = df_profiles[df_profiles["cluster_kmeans"] == cluster_id]["Entity"].sort_values().tolist()

						st.markdown(f"""
						<div style='display:flex; gap:16px; margin-bottom:20px; flex-wrap:wrap;'>
								<div class='metric-card' style='flex:1; min-width:140px;'>
										<h3>Jumlah Negara</h3>
										<div class='val'>{len(countries)}</div>
										<div class='sub'>dari 176 total</div>
								</div>
								<div class='metric-card' style='flex:1; min-width:140px;'>
										<h3>Proporsi</h3>
										<div class='val'>{len(countries)/176*100:.0f}%</div>
										<div class='sub'>dari keseluruhan dataset</div>
								</div>
						</div>
						""", unsafe_allow_html=True)

						# Search
						search = st.text_input("Cari negara", key=f"search_{cluster_id}", placeholder="Ketik nama negara...")
						filtered = [c for c in countries if search.lower() in c.lower()] if search else countries

						# Display as tags
						tags_html = "".join(f"<span class='country-tag-{cluster_id}'>{c}</span>" for c in filtered)
						st.markdown(f"<div style='margin-top:8px; line-height:2;'>{tags_html}</div>", unsafe_allow_html=True)

						st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

						# Profile table
						st.subheader('Profil detail negara:')
						cluster_data = df_profiles[df_profiles["cluster_kmeans"] == cluster_id][["Entity"] + features].copy()
						cluster_data.columns = ["Negara"] + [FEATURE_CONFIG[f]["label"] for f in features]
						if search:
								cluster_data = cluster_data[cluster_data["Negara"].str.lower().str.contains(search.lower())]
						st.dataframe(
								cluster_data.set_index("Negara").style.format("{:.2f}"),
								use_container_width=True,
								height=400,
						)

elif "Tentang" in page:
	st.title('Tentang Model')
   
	tab_method, tab_metrics= st.tabs(["Metodologi", "Metrik Evaluasi"])

	with tab_method:
		c1, c2, c3, c4 = st.columns(4)
		c1.metric("Jumlah Negara", 176)
		c2.metric("Jumlah Cluster", 2)
		c3.metric("Silhouette Score", f"{0.4657}")
		c4.metric("Fitur Dipakai", 5)
		st.divider()

		st.subheader('Dataset')
		st.markdown("""
			**Global Data on Sustainable Energy** [Global Data on Sustainable Energy](https://www.kaggle.com/datasets/anshtanwar/global-data-on-sustainable-energy) (Kaggle)
			  \n176 negara · Tahun 2000–2020 · Diagregasi rata-rata 3 tahun terakhir per negara
			  """)
		st.divider()
		st.subheader('Fitur Yang Digunakan')
		for feat, cfg in FEATURE_CONFIG.items():
				st.markdown(f"""
				<div style='border:1px solid lightgray; border-radius:8px; padding:12px 16px; margin-bottom:8px; display:flex; justify-content:space-between; align-items:center;'>
						<div>
								<div style = 'font-weight: 700'>{cfg["label"]}</div>
								<div>{cfg["help"]}</div>
						</div>
						<div style='font-size:11px;padding:3px 8px; border-radius:4px;'>{cfg["unit"]}</div>
				</div>
				""", unsafe_allow_html=True)

		st.markdown("""
		<div style='margin-top:24px;'>
		<div class='section-title' style='margin-bottom:12px'>Algoritma</div>
		<div class='insight-box' style='margin-bottom:8px;'>
				<p><strong>K-Means (Model Utama)</strong> — Digunakan untuk prediksi. k=2 dipilih berdasarkan Silhouette Score tertinggi dari pencarian k=2 hingga k=10. n_init=50 untuk menghindari local optimum.</p>
		</div>
		<div class='insight-box green' style='margin-bottom:8px;'>
				<p><strong>Hierarchical Clustering (Ward Linkage)</strong> — Sebagai pembanding. Ward linkage dipilih karena menghasilkan cluster yang lebih compact dibanding average linkage.</p>
		</div>
		<div class='insight-box orange' style='margin-bottom:8px;'>
				<p><strong>DBSCAN*</strong> — Sebagai deteksi anomali/outlier, bukan model prediksi utama. Negara dengan profil energi yang terlalu unik (tidak masuk neighborhood manapun dengan eps & min_samples yang ditentukan) ditandai sebagai noise, bukan dipaksa masuk ke salah satu cluster.</p>
		</div>
		</div>
		""", unsafe_allow_html=True)

	with tab_metrics:
		st.subheader("Perbandingan Algoritma Clustering")
		st.markdown(
        "Tiga algoritma dibandingkan dengan k yang sama (k=2 untuk K-Means & Hierarchical). "
        "DBSCAN dievaluasi terpisah karena sebagian besar titik dianggap *noise*."
    )

		metrics = {
				"K-Means": {"Silhouette": 0.4657, "DBI": 0.9353, "CHI": 133.93 , "Noise (%)":	0},
				"Hierarchical": {"Silhouette": 0.4432, "DBI": 0.9821, "CHI": 118.45 , "Noise (%)":	0},
				"DBSCAN*":	{"Silhouette": 0.5417, "DBI":	0.5886, "CHI":26.7445, "Noise (%)":	92.6}
		}

		df = pd.DataFrame(metrics).T

		def highlight(row):
			styles = [""] * len(row)
			return styles

		styled = (
				df.style
				.highlight_max(subset=["Silhouette", "CHI"], color="rgba(37, 235, 73, 0.5)")
				.highlight_min(subset=["DBI"], color="rgba(37, 235, 73, 0.5)")
		)

		st.dataframe(styled, use_container_width=True)

		st.markdown(
        "**Kriteria:** Silhouette & CHI semakin tinggi semakin baik; DBI semakin rendah semakin baik."
    )
		st.divider()

		st.subheader("Mengapa K-Means Dipilih sebagai Model Final?")
		st.markdown(
        "- Menggunakan **seluruh** data negara dalam proses clustering, hasil lebih representatif.\n"
        f"- DBSCAN menganggap **92.6%** "
        "data sebagai noise — sebagian besar negara tidak masuk cluster manapun.\n"
        "- DBI K-Means lebih rendah dan CHI lebih tinggi dibanding Hierarchical Clustering.\n"
        "- Cluster yang terbentuk lebih mudah diinterpretasikan sesuai tujuan riset."
    )

		with st.expander("Lihat negara yang dianggap anomali oleh DBSCAN"):
			st.write(f"{len(noise_countries)} negara dengan profil energi terlalu unik untuk dikelompokkan:")
			tags_html = "".join(f"<span class='country-tag-2'>{c}</span>" for c in noise_countries)
			st.markdown(f"<div style='margin-top:8px; line-height:2;'>{tags_html}</div>", unsafe_allow_html=True)