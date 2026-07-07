# Prediksi Calories Makanan Indonesia

Project machine learning untuk memprediksi kandungan **calories** suatu makanan berdasarkan komposisi gizinya (**protein, fat, carbohydrate**), menggunakan dataset 1346 makanan Indonesia.

## Struktur File

| File | Fungsi |
|---|---|
| `nutrition.csv` | Dataset mentah (id, calories, proteins, fat, carbohydrate, name, image) |
| `prediksi_calories.py` | Script utama: EDA, training & perbandingan 3 model, cross-validation, visualisasi, sekaligus menyimpan model final |
| `app.py` | Aplikasi web (Streamlit) untuk prediksi interaktif |
| `model_rf.pkl` | Model Random Forest yang sudah dilatih (siap pakai) |
| `scaler.pkl` | StandardScaler yang sudah di-fit (untuk keperluan model linear) |
| `regression_viz.png` | Grafik Actual vs Predicted & Residual Plot |
| `requirements.txt` | Daftar library yang dibutuhkan |

## Instalasi

```bash
pip install -r requirements.txt
```

## Cara Menjalankan

### 1. Training model (opsional — model sudah tersedia di `model_rf.pkl`)

```bash
python3 prediksi_calories.py
```

Script ini akan:
- Membaca dan memeriksa kualitas data (`nutrition.csv`)
- Membandingkan performa Linear Regression, Random Forest, dan Gradient Boosting
- Melakukan 5-fold cross-validation
- Menyimpan grafik evaluasi ke `regression_viz.png`
- Melatih ulang model final dengan seluruh data dan menyimpannya ke `model_rf.pkl` + `scaler.pkl`

### 2. Menjalankan aplikasi web

```bash
streamlit run app.py
```

Buka browser ke `http://localhost:8501`, masukkan nilai protein/fat/karbohidrat, lalu klik **Prediksi Calories**.

> Pastikan `app.py`, `model_rf.pkl`, dan `scaler.pkl` berada dalam satu folder yang sama.

## Hasil Evaluasi Model

Dievaluasi pada 20% data uji (270 baris), model tidak pernah melihat data ini saat training:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 40.17 | 64.66 | 0.844 |
| **Random Forest** | **20.37** | **60.94** | **0.861** |
| Gradient Boosting | 21.06 | 61.25 | 0.860 |

**5-fold Cross-Validation (R²):**

| Model | R² rata-rata | Std |
|---|---|---|
| Linear Regression | 0.789 | 0.106 |
| Random Forest | 0.831 | 0.181 |
| Gradient Boosting | 0.786 | 0.238 |

Random Forest dipilih sebagai model final karena performa paling konsisten di kedua metode evaluasi.

**Feature importance (Random Forest):** fat (56%) > carbohydrate (35%) > proteins (9%) — sejalan dengan densitas energi masing-masing makronutrien (fat = 9 kkal/g, protein & karbohidrat = 4 kkal/g).

## Catatan & Limitasi

- Kolom `calories` pada dataset asli memiliki korelasi 0.886 terhadap hasil perhitungan rumus Atwater (`protein×4 + fat×9 + carb×4`) — cukup konsisten, namun ditemukan beberapa baris dengan selisih besar (misal "Pilus", "Kerupuk Udang berpati") yang kemungkinan salah input pada kolom komposisi gizinya. Baris-baris ini menyumbang sebagian besar error model (RMSE > MAE).
- Model saat ini hanya menggunakan 3 fitur numerik (protein, fat, karbohidrat) — belum memanfaatkan kolom `name` sebagai fitur tambahan.
- Cross-validation menunjukkan standar deviasi R² yang cukup besar, terutama pada Gradient Boosting, mengindikasikan sensitivitas terhadap outlier di data.
