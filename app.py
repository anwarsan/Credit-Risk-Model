
import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. Load otak model
model = pickle.load(open('data/credit_model_risk.pkl', 'rb'))
scaler = pickle.load(open('data/scaler.pkl', 'rb'))
features = pickle.load(open('data/features.pkl', 'rb'))

st.title("🛡️ Sistem Penilaian Risiko Kredit JULO")
st.markdown("---")

# 2. Input Form (Sesuai temuan penting di Assignment 2 & 5)
st.sidebar.header("Data Input Pelanggan")

def user_input_features():
    # Fitur paling berpengaruh menurut analisis Anda [cite: 247, 249]
    dpd_p1 = st.sidebar.number_input("Days Past Due (DPD)", 0, 365, 0)
    customer_age = st.sidebar.slider("Usia Pelanggan", 18, 70, 25)
    outstanding_balance = st.sidebar.number_input("Outstanding Balance (Rp)", 0)
    paid_amount_p1 = st.sidebar.number_input("Total Terbayar (Rp)", 0)
    installment_amount = st.sidebar.number_input("Besar Cicilan (Rp)", 0)
    
    data = {
        'dpd_p1': dpd_p1,
        'customer_age': customer_age,
        'outstanding_balance': outstanding_balance,
        'paid_amount_p1': paid_amount_p1,
        'installment_amount': installment_amount
    }
    return pd.DataFrame(data, index=[0])

df_input = user_input_features()

# 3. Proses Prediksi
st.subheader("Profil Data untuk Prediksi")
st.write(df_input)

if st.button("Prediksi Status"):
    # Scaling data input agar sesuai dengan data training [cite: 245]
    scaled_data = scaler.transform(df_input)
    
    # Hasil Prediksi
    prediction = model.predict(scaled_data)
    prediction_proba = model.predict_proba(scaled_data)

    st.markdown("---")
    if prediction[0] == 0: # GOOD [cite: 108]
        st.success("✅ HASIL: **GOOD CUSTOMER**")
        st.write(f"Probabilitas risiko rendah: {prediction_proba[0][0]:.2%}")
    else: # BAD [cite: 108]
        st.error("⚠️ HASIL: **BAD CUSTOMER**")
        st.write(f"Probabilitas risiko tinggi: {prediction_proba[0][1]:.2%}")
        st.warning("Insight: Pelanggan di bawah 30 tahun memiliki risiko default lebih tinggi[cite: 149, 218].")
