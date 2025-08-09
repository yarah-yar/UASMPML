import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


st.title("📊 Prediksi Profitabilitas Menu Restoran")
st.write("Masukkan detail menu untuk memprediksi apakah item tersebut **High Profit** atau **Low Profit**")

# ===== Input User =====
price = st.number_input("💲 Harga Jual (Price)", min_value=0.0, step=0.01)
cost = st.number_input("💰 Biaya Produksi (Cost)", min_value=0.0, step=0.01)
quantity_sold = st.number_input("📦 Jumlah Terjual (Quantity Sold)", min_value=0, step=1)

# Kategori menu sesuai dataset training
category_options = ["Appetizer", "Main Course", "Dessert", "Beverage"]
category = st.selectbox("🍽️ Kategori Menu", category_options)

# ===== Proses Input =====
# One-hot encoding manual (harus sama urutan & nama kolomnya seperti saat training)
category_encoded = {f"Category_{cat}": 0 for cat in category_options}
category_encoded[f"Category_{category}"] = 1

# Buat dataframe input sesuai kolom training
input_df = pd.DataFrame([{
    "Price": price,
    "Cost": cost,
    "Quantity_Sold": quantity_sold,
    **category_encoded
}])

# Pastikan urutan kolom sama seperti pada training
model_features = scaler.feature_names_in_  # Ambil urutan kolom dari scaler saat training
input_df = input_df.reindex(columns=model_features, fill_value=0)

# Scaling
input_scaled = scaler.transform(input_df)

# ===== Prediksi =====
if st.button("Prediksi Profitabilitas"):
    prediction = model.predict(input_scaled)[0]
    if prediction == 1 or prediction == "High Profit":
        st.success("✅ Item menu ini diprediksi **High Profit**")
    else:
        st.error("❌ Item menu ini diprediksi **Low Profit**")
