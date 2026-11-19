[README (2).md](https://github.com/user-attachments/files/35091780/README.2.md)
# xgboost-soil-crop-ai# 🌱 XGBoost Soil-to-Crop AI: Riset Klasifikasi Kesesuaian Tanaman Berdasarkan Kadar Nutrisi Tanah

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-10B981)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/GUI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

Proyek riset mata kuliah **Kecerdasan Buatan (Artificial Intelligence)** yang menerapkan algoritma **XGBoost (*Extreme Gradient Boosting*)**, *Agronomic Feature Engineering* (rasio keseimbangan NPK & deviasi pH), *Ablation Study*, dan **SHAP (*Explainable AI*)** untuk merekomendasikan 22 komoditas tanaman pertanian berdasarkan kadar kimia tanah dan kondisi agroklimat.

Repositori ini juga dilengkapi dengan **GUI Real-Time Interaktif** (berbasis **Streamlit** maupun **Web HTML/GitHub Pages**) agar pengguna dapat mengamati jalur keputusan pohon XGBoost, skor *Log-Odds Margin*, distribusi probabilitas *Softmax*, dan kontribusi fitur secara langsung.

---

## 👨‍💻 Identitas Penulis

| Keterangan | Detail |
| :--- | :--- |
| **Nama** | Maulana Dwi Febrian Arajak |
| **NIM** | 25051204186 |
| **Program Studi** | S1 Teknik Informatika — Universitas Negeri Surabaya (UNESA) |
| **Kelas** | TIC25 |
| **Mata Kuliah** | Kecerdasan Buatan (Artificial Intelligence) |

---

## 📊 Dataset & Rekayasa Fitur (*Feature Engineering*)

Dataset utama bersumber dari Kaggle: **[Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)** dengan karakteristik:
* **2.200 sampel** terdistribusi seimbang pada **22 kelas tanaman** (100 sampel per komoditas, 0 *missing values*).
* **Parameter Kimia Tanah**: Nitrogen (`N`), Fosfor (`P`), Kalium (`K`), dan Keasaman Tanah (`ph`).
* **Parameter Agroklimat**: Suhu (`temperature`), Kelembapan (`humidity`), dan Curah Hujan (`rainfall`).
* **8 Fitur Turunan Agronomi**: `Total_NPK`, `N_to_P_ratio`, `N_to_K_ratio`, `P_to_K_ratio`, `N_share`, `P_share`, `K_share`, dan `pH_deviation_from_neutral` ($|	ext{pH} - 6.5|$).

---

## 🧪 Hasil Eksperimen Riset

### 1. Ablation Study (5-Fold Stratified Cross-Validation)

| Skenario Fitur | Jumlah Fitur | Mean Macro F1-Score | Std Dev |
| :--- | :---: | :---: | :---: |
| **Skenario 1**: Kadar Tanah Murni (`N, P, K, ph`) | 4 | `0.7850` (**78,50%**) | `± 0.0057` |
| **Skenario 2**: Kadar Tanah + Rasio Agronomi NPK | 12 | `0.7856` (**78,56%**) | `± 0.0125` |
| **Skenario 3**: Kadar Tanah + Rasio + Agroklimat | 15 | `0.9863` (**98,63%**) | `± 0.0064` |

> **Temuan Utama:** Mengandalkan unsur kimia tanah saja menghasilkan *F1-Score* **78,50%** karena komoditas seperti *Rice* (Padi) dan *Jute* (Rami) memiliki profil NPK dan pH yang berdekatan. Penggabungan dengan faktor agroklimat (curah hujan, kelembapan, suhu) meningkatkan performa sebesar **+20,07%** menjadi **98,63%**.

### 2. Komparasi Model Klasifikasi (20% Stratified Test Split)

| Algoritma | Test Accuracy | Macro F1-Score |
| :--- | :---: | :---: |
| Decision Tree (Baseline) | `97,50%` | `0.9749` |
| XGBoost (Default) | `97,73%` | `0.9771` |
| Random Forest (200 Trees) | `99,32%` | `0.9932` |

---

## 📈 Visualisasi Exploratory Data Analysis (EDA)

| Korelasi Parameter Tanah & Iklim | Profil Rata-Rata N, P, K per Tanaman |
| :---: | :---: |
| ![Korelasi Parameter Tanah](docs/images/1_korelasi_parameter_tanah_iklim.png) | ![Distribusi NPK](docs/images/2_distribusi_npk_per_tanaman.png) |

| Distribusi Rentang pH Tanah | Peta Klaster Makronutrisi (N vs P & K) |
| :---: | :---: |
| ![Distribusi pH](docs/images/3_distribusi_ph_tanah_per_tanaman.png) | ![Peta Klaster NPK](docs/images/4_peta_klaster_npk_tanaman.png) |

---

## 📁 Struktur Repositori

```text
xgboost-soil-crop-ai/
├── data/
│   ├── Crop_recommendation.csv                  # Dataset asli Kaggle (2.200 baris)
│   ├── Crop_recommendation_engineered.csv       # Dataset + 8 fitur rasio agronomi
│   └── statistik_kadar_tanah_per_tanaman.csv    # Ringkasan mean, std, min, max 22 tanaman
├── docs/
│   └── images/                                  # Grafik EDA resolusi tinggi (300 DPI)
├── notebooks/
│   └── notebook_riset_xgboost_tanah.ipynb       # Jupyter Notebook eksperimen interaktif
├── app_gui_xgboost_realtime.py                  # Aplikasi GUI Python Real-Time (Streamlit)
├── riset_tanah_tanaman_xgboost.py               # Skrip eksperimen lengkap (Ablation + Tuning + SHAP)
├── index.html                                   # Simulator GUI Web Interaktif (Siap GitHub Pages)
├── requirements.txt                             # Daftar dependensi Python
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🚀 Cara Menjalankan Proyek

### 1. Clone Repositori & Instalasi Dependensi
```bash
git clone https://github.com/<username-github-kamu>/xgboost-soil-crop-ai.git
cd xgboost-soil-crop-ai
pip install -r requirements.txt
```

### 2. Menjalankan GUI Real-Time Streamlit (Model XGBoost Asli)
```bash
streamlit run app_gui_xgboost_realtime.py
```
Browser akan terbuka otomatis di `http://localhost:8501` dengan *slider* kadar tanah (`N, P, K, pH`), agroklimat, probabilitas *Softmax*, dan jejak *leaf index* XGBoost.

### 3. Menjalankan Skrip Eksperimen Riset (CLI + Ekspor Grafik)
```bash
python riset_tanah_tanaman_xgboost.py
```

### 4. Demo Web Simulator Instan (GitHub Pages)
File `index.html` di *root* repositori dapat langsung dibuka di browser atau dihosting gratis melalui **Settings → Pages → Deploy from a branch (`main` / `/root`)** di GitHub.

---

## 📜 Lisensi
Proyek ini dirilis di bawah lisensi [MIT](LICENSE) untuk keperluan pembelajaran dan riset akademik.
