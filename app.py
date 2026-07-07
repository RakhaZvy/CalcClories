"""
APLIKASI WEB - Prediksi Calories dari Kandungan Gizi
Jalankan dengan: streamlit run app.py
(pastikan sudah install: pip install streamlit joblib pandas scikit-learn)
"""

import streamlit as st       # library utama buat bikin UI web dari script Python
import joblib                # buat load model & scaler yang sudah disimpan
import pandas as pd
import numpy as np

# =========================================================
# 1. KONFIGURASI HALAMAN (harus di baris paling atas setelah import)
# =========================================================
st.set_page_config(
    page_title="Prediksi Calories Makanan",
    page_icon="🍚",
    layout="centered"
)

# =========================================================
# 2. LOAD MODEL & SCALER (cuma sekali, di-cache biar nggak reload tiap interaksi)
# =========================================================
@st.cache_resource
# decorator ini bikin Streamlit "mengingat" hasil fungsi di bawahnya,
# jadi model cuma di-load 1x pas aplikasi pertama dibuka, bukan tiap
# kali user geser slider (kalau nggak di-cache, bakal lambat banget)
def load_model():
    model = joblib.load('model_rf.pkl')
    return model, scaler

model = load_model()

# =========================================================
# 3. JUDUL & DESKRIPSI HALAMAN
# =========================================================
st.title("🍚 Prediksi Calories Makanan")
st.write(
    "Masukkan kandungan gizi makanan (protein, lemak, karbohidrat), "
    "dan model Machine Learning (Random Forest) akan memprediksi estimasi kalorinya."
)

# =========================================================
# 4. INPUT DARI USER (sidebar atau halaman utama)
# =========================================================
st.subheader("Input Kandungan Gizi (per takaran saji)")

col1, col2, col3 = st.columns(3)
# st.columns(3) membagi layout jadi 3 kolom sejajar, biar input rapi bersebelahan

with col1:
    proteins = st.number_input(
        "Protein (gram)",
        min_value=0.0, max_value=100.0, value=10.0, step=0.5
    )
    # number_input bikin kotak input angka, dengan batas min/max dan nilai default

with col2:
    fat = st.number_input(
        "Fat / Lemak (gram)",
        min_value=0.0, max_value=100.0, value=5.0, step=0.5
    )

with col3:
    carbohydrate = st.number_input(
        "Karbohidrat (gram)",
        min_value=0.0, max_value=650.0, value=20.0, step=0.5
    )

# =========================================================
# 5. TOMBOL PREDIKSI
# =========================================================
if st.button("🔍 Prediksi Calories", type="primary"):
    # Kode di dalam blok ini HANYA jalan ketika tombol diklik

    # Bungkus input jadi DataFrame dengan nama kolom yang SAMA
    # persis dengan yang dipakai waktu training (proteins, fat, carbohydrate)
    input_data = pd.DataFrame({
        'proteins': [proteins],
        'fat': [fat],
        'carbohydrate': [carbohydrate]
    })

    prediksi = model.predict(input_data)[0]
    # model.predict() selalu mengembalikan array (walau cuma 1 input),
    # makanya diambil elemen pertama pakai [0]

    st.success(f"### Estimasi Calories: **{prediksi:.1f} kkal**")

    # -----------------------------------------------------
    # Breakdown kontribusi tiap makronutrien (opsional tapi informatif)
    # -----------------------------------------------------
    st.subheader("Breakdown Kontribusi Energi (rumus Atwater)")
    kalori_protein = proteins * 4
    kalori_fat = fat * 9
    kalori_carb = carbohydrate * 4

    breakdown_df = pd.DataFrame({
        'Sumber': ['Protein', 'Fat', 'Karbohidrat'],
        'Kalori': [kalori_protein, kalori_fat, kalori_carb]
    })
    st.bar_chart(breakdown_df.set_index('Sumber'))
    # bar_chart otomatis bikin grafik batang dari DataFrame, tanpa perlu matplotlib

    st.caption(
        f"Total dari rumus Atwater: {kalori_protein + kalori_fat + kalori_carb:.1f} kkal "
        f"(dibandingkan prediksi model: {prediksi:.1f} kkal)"
    )

# =========================================================
# 6. INFO TAMBAHAN DI SIDEBAR
# =========================================================
st.sidebar.header("Tentang Model")
st.sidebar.write(
    "Model: Random Forest Regressor\n\n"
    "Dilatih dari 1346 data makanan Indonesia "
    "(kolom: proteins, fat, carbohydrate → calories)."
)
st.sidebar.write("R² pada data uji: ~0.86")
