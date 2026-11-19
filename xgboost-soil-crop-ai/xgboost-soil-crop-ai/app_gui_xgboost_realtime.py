"""
Aplikasi GUI Real-Time AI XGBoost (Streamlit)
Riset Klasifikasi Kesesuaian Tanaman Berdasarkan Kadar Nutrisi Tanah (N, P, K, pH) & Agroklimat
Jalankan dengan perintah:
    streamlit run app_gui_xgboost_realtime.py
"""

import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

st.set_page_config(
    page_title="XGBoost Soil-to-Crop AI Real-Time Inspector",
    page_icon="🌱",
    layout="wide",
)

def add_agronomic_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    eps = 1e-5
    data["Total_NPK"] = data["N"] + data["P"] + data["K"]
    data["N_to_P_ratio"] = data["N"] / (data["P"] + eps)
    data["N_to_K_ratio"] = data["N"] / (data["K"] + eps)
    data["P_to_K_ratio"] = data["P"] / (data["K"] + eps)
    data["N_share"] = data["N"] / (data["Total_NPK"] + eps)
    data["P_share"] = data["P"] / (data["Total_NPK"] + eps)
    data["K_share"] = data["K"] / (data["Total_NPK"] + eps)
    data["pH_deviation_from_neutral"] = np.abs(data["ph"] - 6.5)
    return data


@st.cache_resource
def load_and_train_models(csv_path: str = "Crop_recommendation.csv"):
    df = pd.read_csv(csv_path)
    df_feat = add_agronomic_features(df)

    le = LabelEncoder()
    y = le.fit_transform(df_feat["label"])

    soil_only = ["N", "P", "K", "ph"]
    soil_ratios = soil_only + [
        "Total_NPK", "N_to_P_ratio", "N_to_K_ratio", "P_to_K_ratio",
        "N_share", "P_share", "K_share", "pH_deviation_from_neutral",
    ]
    full_cols = soil_ratios + ["temperature", "humidity", "rainfall"]

    scenarios = {
        "Skenario 1: Kadar Tanah Murni (N, P, K, pH)": soil_only,
        "Skenario 2: Kadar Tanah + Rasio Agronomi NPK": soil_ratios,
        "Skenario 3: Lengkap (Tanah + Rasio + Agroklimat)": full_cols,
    }

    trained = {}
    for name, cols in scenarios.items():
        X_train, X_test, y_train, y_test = train_test_split(
            df_feat[cols], y, test_size=0.20, stratify=y, random_state=42
        )
        model = XGBClassifier(
            n_estimators=120,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.85,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        trained[name] = {
            "model": model,
            "cols": cols,
            "acc": accuracy_score(y_test, preds),
            "f1": f1_score(y_test, preds, average="macro"),
            "X_train": X_train,
        }

    return df, le, trained


st.title("🌱 XGBoost Soil-to-Crop AI Real-Time Inspector")
st.caption(
    "Visualisasi langsung cara kerja model XGBoost asli pada 2.200 sampel Crop_recommendation.csv (22 komoditas tanaman)."
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_CANDIDATES = [
    os.path.join(BASE_DIR, "data", "Crop_recommendation.csv"),
    os.path.join(BASE_DIR, "Crop_recommendation.csv"),
    "data/Crop_recommendation.csv",
    "Crop_recommendation.csv",
]
CSV_PATH = next((p for p in CSV_CANDIDATES if os.path.exists(p)), None)
if CSV_PATH is None:
    st.error("File `Crop_recommendation.csv` tidak ditemukan di folder `data/` maupun root proyek.")
    st.stop()

df, le, trained_models = load_and_train_models(CSV_PATH)

st.sidebar.header("⚙️ Kontrol Simulasi AI")
scenario_name = st.sidebar.selectbox(
    "Pilih Mode Skenario Fitur (Ablation):",
    list(trained_models.keys()),
    index=2,
)
active_bundle = trained_models[scenario_name]
model = active_bundle["model"]
active_cols = active_bundle["cols"]

st.sidebar.markdown(
    f"**Performa Model Aktif (Test 20%):**  \n"
    f"Accuracy: `{active_bundle['acc']*100:.2f}%` | Macro F1: `{active_bundle['f1']*100:.2f}%`"
)
st.sidebar.divider()

crop_list = sorted(df["label"].unique())
preset_crop = st.sidebar.selectbox("Muat Rata-Rata Sampel Asli Tanaman:", ["(Manual)"] + crop_list)
if preset_crop != "(Manual)":
    ref = df[df["label"] == preset_crop].mean(numeric_only=True)
    dN, dP, dK, dPh = int(ref["N"]), int(ref["P"]), int(ref["K"]), float(round(ref["ph"], 2))
    dTemp, dHum, dRain = float(round(ref["temperature"], 1)), float(round(ref["humidity"], 1)), float(round(ref["rainfall"], 1))
else:
    dN, dP, dK, dPh = 80, 48, 40, 6.43
    dTemp, dHum, dRain = 23.7, 82.3, 236.2

st.sidebar.subheader("1. Sensor Kadar Kimia Tanah")
val_N = st.sidebar.slider("Nitrogen (N) [mg/kg]", 0, 140, dN)
val_P = st.sidebar.slider("Fosfor (P) [mg/kg]", 5, 145, dP)
val_K = st.sidebar.slider("Kalium (K) [mg/kg]", 5, 205, dK)
val_ph = st.sidebar.slider("Keasaman Tanah (pH)", 3.5, 9.9, dPh, step=0.05)

st.sidebar.subheader("2. Sensor Agroklimat")
val_temp = st.sidebar.slider("Suhu Udara (°C)", 8.8, 43.7, dTemp, step=0.2)
val_hum = st.sidebar.slider("Kelembapan Udara (%)", 14.0, 100.0, dHum, step=0.5)
val_rain = st.sidebar.slider("Curah Hujan (mm)", 20.0, 299.0, dRain, step=1.0)

input_raw = pd.DataFrame([{
    "N": val_N, "P": val_P, "K": val_K, "ph": val_ph,
    "temperature": val_temp, "humidity": val_hum, "rainfall": val_rain
}])
input_eng = add_agronomic_features(input_raw)
X_infer = input_eng[active_cols]

t0 = time.perf_counter()
probs = model.predict_proba(X_infer)[0]
latency_ms = (time.perf_counter() - t0) * 1000

top_indices = np.argsort(probs)[::-1][:5]
top_crops = le.inverse_transform(top_indices)
top_probs = probs[top_indices] * 100

col_main, col_prob = st.columns([1.1, 1.4])

with col_main:
    st.subheader("🎯 Keputusan Utama Model XGBoost")
    st.success(f"### **{top_crops[0].upper()}** — Keyakinan: **{top_probs[0]:.2f}%**")
    st.caption(f"Waktu inferensi real-time: `{latency_ms:.2f} ms` | Total Pohon Ensemble: `120 Trees`")

    m1, m2, m3 = st.columns(3)
    m1.metric("Total NPK", f"{input_eng['Total_NPK'].iloc[0]:.0f} mg/kg")
    m2.metric("Rasio N / P", f"{input_eng['N_to_P_ratio'].iloc[0]:.2f}")
    m3.metric("|pH - 6.5|", f"{input_eng['pH_deviation_from_neutral'].iloc[0]:.2f}")

with col_prob:
    st.subheader("📊 Distribusi Probabilitas Softmax (Top-5 Tanaman)")
    prob_df = pd.DataFrame({
        "Tanaman": [c.upper() for c in top_crops],
        "Probabilitas (%)": top_probs,
    })
    fig_p, ax_p = plt.subplots(figsize=(6, 3.2))
    sns.barplot(data=prob_df, x="Probabilitas (%)", y="Tanaman", palette="viridis", ax=ax_p)
    ax_p.set_xlim(0, 105)
    for i, v in enumerate(top_probs):
        ax_p.text(v + 1.2, i, f"{v:.1f}%", va="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig_p)
    plt.close(fig_p)

st.divider()

col_tree, col_imp = st.columns(2)

with col_tree:
    st.subheader("🌳 Jejak Daun (Leaf Index) & Kontribusi Pohon XGBoost")
    leaf_indices = model.apply(X_infer)[0][:5]
    raw_margins = model.predict(X_infer, output_margin=True)[0]
    top_margin = raw_margins[top_indices[0]]
    runner_margin = raw_margins[top_indices[1]]

    st.write(
        f"Sampel tanahmu melewati jalur percabangan spesifik di setiap pohon (*Gradient Boosted Trees*). "
        f"Skor log-odds (*raw margin*) untuk **{top_crops[0].upper()}** adalah **`{top_margin:+.3f}`** "
        f"(unggul `{top_margin - runner_margin:.3f}` dari kandidat #2 **{top_crops[1].upper()}**)."
    )
    leaf_df = pd.DataFrame({
        "Pohon Boosting": [f"Tree #{i+1}" for i in range(5)],
        "ID Leaf Aktif (Sampel Ini)": [int(x) for x in leaf_indices],
        "Status Jalur": ["Aktif (Evaluasi Split)"] * 5,
    })
    st.dataframe(leaf_df, use_container_width=True, hide_index=True)

with col_imp:
    st.subheader("🔍 Kontribusi Global Fitur (XGBoost Gain Importance)")
    imp_df = pd.DataFrame({
        "Fitur": active_cols,
        "Gain Importance": model.feature_importances_,
    }).sort_values("Gain Importance", ascending=False).head(10)

    fig_i, ax_i = plt.subplots(figsize=(6, 3.6))
    sns.barplot(data=imp_df, x="Gain Importance", y="Fitur", palette="crest", ax=ax_i)
    plt.tight_layout()
    st.pyplot(fig_i)
    plt.close(fig_i)
