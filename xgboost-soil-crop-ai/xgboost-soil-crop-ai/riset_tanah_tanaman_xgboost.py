"""
Riset Mata Kuliah Kecerdasan Buatan (AI)
Topik   : Klasifikasi Kesesuaian Tanaman Berdasarkan Kadar Nutrisi Tanah (N, P, K, pH)
          dan Parameter Agroklimat Menggunakan Optimasi XGBoost & Analisis SHAP
Dataset : Kaggle Crop Recommendation Dataset
          https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# =====================================================================
# 1. PEMUATAN DATA & VALIDASI STRUKTUR (KAGGLE CROP RECOMMENDATION)
# =====================================================================
def load_dataset(csv_path: str = None) -> pd.DataFrame:
    """
    Memuat dataset Crop_recommendation.csv dari Kaggle.
    Jika file belum diunduh ke direktori kerja, fungsi akan memberi panduan unduh.
    """
    if csv_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        for candidate in [os.path.join(base_dir, "data", "Crop_recommendation.csv"), os.path.join(base_dir, "Crop_recommendation.csv"), "data/Crop_recommendation.csv", "Crop_recommendation.csv"]:
            if os.path.exists(candidate):
                csv_path = candidate
                break
        else:
            csv_path = "data/Crop_recommendation.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"File '{csv_path}' belum ditemukan di folder kerja.\n"
            "Unduh dari Kaggle: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset\n"
            "Lalu simpan sebagai 'Crop_recommendation.csv' di folder yang sama dengan script ini."
        )
    df = pd.read_csv(csv_path)
    print(f"[INFO] Dataset berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom.")
    print(f"[INFO] Daftar Kelas Tanaman ({df['label'].nunique()} jenis): { sorted(df['label'].unique()) }")
    return df


# =====================================================================
# 2. FEATURE ENGINEERING BERBASIS AGRONOMI (KADAR TANAH)
# =====================================================================
def add_soil_agronomic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menambahkan fitur turunan kimia tanah untuk menguji hipotesis riset:
    Apakah rasio keseimbangan nutrisi NPK meningkatkan akurasi & pemisahan kelas?
    """
    data = df.copy()
    eps = 1e-5  # Mencegah pembagian dengan nol

    # Total makronutrisi utama di dalam tanah
    data["Total_NPK"] = data["N"] + data["P"] + data["K"]

    # Rasio keseimbangan antar-nutrisi tanah
    data["N_to_P_ratio"] = data["N"] / (data["P"] + eps)
    data["N_to_K_ratio"] = data["N"] / (data["K"] + eps)
    data["P_to_K_ratio"] = data["P"] / (data["K"] + eps)

    # Proporsi masing-masing unsur terhadap total NPK
    data["N_share"] = data["N"] / (data["Total_NPK"] + eps)
    data["P_share"] = data["P"] / (data["Total_NPK"] + eps)
    data["K_share"] = data["K"] / (data["Total_NPK"] + eps)

    # Indeks deviasi keasaman tanah dari pH netral ideal (6.5)
    data["pH_deviation_from_neutral"] = np.abs(data["ph"] - 6.5)

    return data


# =====================================================================
# 3. KOMPARASI ABLATION STUDY (FITUR MURNI TANAH vs TANAH + IKLIM)
# =====================================================================
def run_ablation_study(df_engineered: pd.DataFrame, y_encoded: np.ndarray):
    """
    Membandingkan performa XGBoost pada 3 skenario fitur untuk bahan pembahasan laporan:
    1. Hanya Kadar Kimia Tanah Murni (N, P, K, ph)
    2. Kadar Kimia Tanah + Rasio Agronomi NPK
    3. Gabungan Lengkap (Kadar Tanah + Rasio NPK + Iklim/Cuaca)
    """
    soil_only_cols = ["N", "P", "K", "ph"]
    soil_engineered_cols = [
        "N", "P", "K", "ph",
        "Total_NPK", "N_to_P_ratio", "N_to_K_ratio", "P_to_K_ratio",
        "N_share", "P_share", "K_share", "pH_deviation_from_neutral",
    ]
    full_cols = soil_engineered_cols + ["temperature", "humidity", "rainfall"]

    scenarios = {
        "Skenario 1: Kadar Tanah Murni (N, P, K, pH)": soil_only_cols,
        "Skenario 2: Kadar Tanah + Rasio Agronomi": soil_engineered_cols,
        "Skenario 3: Kadar Tanah + Rasio + Agroklimat": full_cols,
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    ablation_results = []

    print("\n" + "=" * 70)
    print("HASIL ABLATION STUDY (5-FOLD STRATIFIED CROSS-VALIDATION)")
    print("=" * 70)

    for name, cols in scenarios.items():
        X_sub = df_engineered[cols]
        model = XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )
        scores = cross_val_score(model, X_sub, y_encoded, cv=skf, scoring="f1_macro")
        ablation_results.append({
            "Skenario Fitur": name,
            "Jumlah Fitur": len(cols),
            "Mean Macro F1-Score": scores.mean(),
            "Std Dev": scores.std(),
        })
        print(f"{name:48s} | F1-Macro: {scores.mean():.4f} (+/- {scores.std():.4f})")

    return pd.DataFrame(ablation_results)


# =====================================================================
# 4. KOMPARASI MODEL & HYPERPARAMETER TUNING XGBOOST
# =====================================================================
def train_and_evaluate_models(X_train, X_test, y_train, y_test, label_encoder):
    """
    Membandingkan Decision Tree, Random Forest, Baseline XGBoost, dan Tuned XGBoost.
    """
    models = {
        "Decision Tree (Baseline)": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "XGBoost (Default)": XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        ),
    }

    summary = []
    print("\n" + "=" * 70)
    print("KOMPARASI MODEL KLASIFIKASI TANAMAN")
    print("=" * 70)

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        summary.append({"Model": name, "Accuracy": acc, "Macro F1-Score": f1})
        print(f"{name:30s} | Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")

    # Hyperparameter Tuning XGBoost menggunakan RandomizedSearchCV
    param_dist = {
        "n_estimators": [150, 250, 350],
        "max_depth": [4, 6, 8, 10],
        "learning_rate": [0.03, 0.05, 0.1, 0.15],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "reg_alpha": [0.0, 0.1, 0.5],
        "reg_lambda": [1.0, 1.5, 2.0],
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator=XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        ),
        param_distributions=param_dist,
        n_iter=15,
        scoring="f1_macro",
        cv=skf,
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best_xgb = search.best_estimator_

    tuned_preds = best_xgb.predict(X_test)
    tuned_acc = accuracy_score(y_test, tuned_preds)
    tuned_f1 = f1_score(y_test, tuned_preds, average="macro")
    summary.append({
        "Model": "XGBoost (Tuned)",
        "Accuracy": tuned_acc,
        "Macro F1-Score": tuned_f1,
    })

    print(f"{'XGBoost (Tuned)':30s} | Accuracy: {tuned_acc:.4f} | Macro F1: {tuned_f1:.4f}")
    print(f"\n[INFO] Hyperparameter Terbaik XGBoost:\n{search.best_params_}")

    print("\n" + "=" * 70)
    print("LAPORAN KLASIFIKASI PER KOMODITAS TANAMAN (XGBOOST TUNED)")
    print("=" * 70)
    print(
        classification_report(
            y_test,
            tuned_preds,
            target_names=label_encoder.classes_,
            digits=4,
        )
    )

    return best_xgb, pd.DataFrame(summary), tuned_preds


# =====================================================================
# 5. VISUALISASI FEATURE IMPORTANCE & CONFUSION MATRIX UNTUK LAPORAN
# =====================================================================
def save_research_plots(best_xgb, X_train, y_test, y_pred, label_encoder, output_dir="output_riset"):
    """
    Menyimpan grafik Confusion Matrix dan Feature Importance (Gain) resolusi tinggi
    untuk dimasukkan ke dalam laporan / paper mata kuliah AI.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(14, 11))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
    )
    plt.title("Confusion Matrix - Rekomendasi Tanaman Berdasarkan Kadar Tanah (XGBoost)")
    plt.xlabel("Prediksi Tanaman")
    plt.ylabel("Tanaman Aktual")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "1_confusion_matrix_xgboost.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # 2. Plot Feature Importance berdasarkan Gain XGBoost
    importance_df = pd.DataFrame({
        "Fitur": X_train.columns,
        "Importance_Gain": best_xgb.feature_importances_,
    }).sort_values(by="Importance_Gain", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x="Importance_Gain", y="Fitur", palette="viridis")
    plt.title("Kontribusi Fitur Kadar Tanah & Lingkungan pada Keputusan XGBoost")
    plt.xlabel("XGBoost Feature Importance (Gain)")
    plt.ylabel("Parameter Tanah & Lingkungan")
    plt.tight_layout()
    fi_path = os.path.join(output_dir, "2_feature_importance_tanah.png")
    plt.savefig(fi_path, dpi=300)
    plt.close()

    # 3. Plot SHAP Values (Opsional jika library shap terinstal)
    try:
        import shap
        explainer = shap.TreeExplainer(best_xgb)
        shap_values = explainer.shap_values(X_train.iloc[:300])
        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_values, X_train.iloc[:300], plot_type="bar", show=False)
        plt.tight_layout()
        shap_path = os.path.join(output_dir, "3_shap_summary_plot.png")
        plt.savefig(shap_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Grafik SHAP berhasil disimpan di: {shap_path}")
    except ImportError:
        print("[INFO] Library 'shap' belum terinstal (jalankan `pip install shap` jika ingin grafik SHAP).")

    print(f"[INFO] Grafik laporan riset tersimpan di folder: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    # Pastikan file Crop_recommendation.csv sudah diunduh dari Kaggle
    df_raw = load_dataset()
    df_feat = add_soil_agronomic_features(df_raw)

    le = LabelEncoder()
    y_all = le.fit_transform(df_feat["label"])
    X_all = df_feat.drop(columns=["label"])

    # Jalankan Ablation Study (Fitur Tanah Murni vs Tanah + Rasio vs Lengkap)
    ablation_df = run_ablation_study(df_feat, y_all)

    # Split Data 80:20 secara Stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.20, stratify=y_all, random_state=42
    )

    # Latih & Evaluasi Model
    best_model, comparison_df, y_pred = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, le
    )

    # Simpan Visualisasi Riset
    save_research_plots(best_model, X_train, y_test, y_pred, le)