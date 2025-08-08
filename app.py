!pip install streamlit scikit-learn pandas numpy joblib --quiet

import joblib

# Simpan model terbaik hasil training (misalnya best_rf dari Tahap 3)
joblib.dump(best_rf, "model_restaurant_profit.pkl")

# Buat file app.py untuk Streamlit
app_code = """
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load('model_restaurant_profit.pkl')

st.title("Prediksi Profitabilitas Menu Restoran")
st.write("Masukkan informasi menu untuk memprediksi profitabilitas (High Profit / Low Profit)")

# Form input
price = st.number_input("Harga Jual (Price)", min_value=0.0, step=0.01)
cost = st.number_input("Biaya Produksi (Cost)", min_value=0.0, step=0.01)
quantity_sold = st.number_input("Jumlah Terjual (Quantity Sold)", min_value=0, step=1)

# Kategori menu (ganti sesuai kategori di dataset)
category = st.selectbox("Kategori Menu", ["Appetizer", "Main Course", "Dessert", "Beverage"])

# One-hot encoding manual sesuai training
category_encoded = {
    "Category_Appetizer": 0,
    "Category_Beverage": 0,
    "Category_Dessert": 0,
    "Category_Main Course": 0
}
if category == "Appetizer":
    category_encoded["Category_Appetizer"] = 1
elif category == "Beverage":
    category_encoded["Category_Beverage"] = 1
elif category == "Dessert":
    category_encoded["Category_Dessert"] = 1
elif category == "Main Course":
    category_encoded["Category_Main Course"] = 1

# Gabungkan fitur numerik & kategorikal
input_data = pd.DataFrame([{
    "Price": price,
    "Cost": cost,
    "Quantity_Sold": quantity_sold,
    **category_encoded
}])

# Prediksi
if st.button("Prediksi Profitabilitas"):
    prediction = model.predict(input_data)[0]
    st.subheader(f"Hasil Prediksi: {prediction}")
"""

with open("app.py", "w") as f:
    f.write(app_code)

# Buat requirements.txt
with open("requirements.txt", "w") as f:
    f.write("streamlit\nscikit-learn\nnumpy\npandas\njoblib\n")

print("File Streamlit app.py dan requirements.txt sudah dibuat.")
