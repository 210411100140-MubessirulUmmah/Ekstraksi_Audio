import streamlit as st
import numpy as np
import pickle
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(
    page_title="PreProcesing Data",
    page_icon="🧹",
    layout="wide"
)

st.title("🧹 PreProcesing Data")

scaler = st.radio(
"Metode normalisasi data",
('Tanpa Normalisasi Data', 'Zscore Scaler', 'MinMax Scaler'))
if scaler == 'Tanpa Normalisasi Data':
    st.title("Dataset Tanpa Preprocessing : ")
    df = pd.read_csv('hasil_statistik2.csv')
    df
elif scaler == 'Zscore Scaler':
    st.title('Hasil Normalisasi Menggunakan Z-score')

    # Membaca data dari file CSV
    df = pd.read_csv('hasil_statistik2.csv')

    # Memisahkan kolom target (label) dari kolom fitur
    X = df.drop(columns=['Label'])  # Kolom fitur
    y = df['Label']  # Kolom target

    # Membagi data menjadi data training dan data testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalisasi Z-score pada data training
    zscore_scaler = StandardScaler()
    X_train_zscore = zscore_scaler.fit_transform(X_train)

    # Membuat DataFrame dari data hasil normalisasi Z-score
    normalized_data_zscore = pd.DataFrame(X_train_zscore, columns=X_train.columns)

    # Menambahkan kolom target (label) ke DataFrame hasil normalisasi Z-score
    normalized_data_zscore['Label'] = y_train

    # Menampilkan data hasil normalisasi Z-score
    print("Data Hasil Normalisasi Z-score:")
    normalized_data_zscore

elif scaler == 'MinMax Scaler':
    st.title('Hasil Normalisasi Menggunakan Min-Max')

    # Membaca data dari file CSV
    df = pd.read_csv('hasil_statistik2.csv')

    # Memisahkan kolom target (label) dari kolom fitur
    X = df.drop(columns=['Label'])  # Kolom fitur
    y = df['Label']  # Kolom target

    # Membagi data menjadi data training dan data testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalisasi Min-Max Scaling pada data training
    minmax_scaler = MinMaxScaler()
    X_train_minmax = minmax_scaler.fit_transform(X_train)

    # Membuat DataFrame dari data hasil normalisasi Min-Max Scaling
    normalized_data_minmax = pd.DataFrame(X_train_minmax, columns=X_train.columns)

    # Menambahkan kolom target (label) ke DataFrame hasil normalisasi Min-Max Scaling
    normalized_data_minmax['Label'] = y_train

    # Menampilkan data hasil normalisasi Min-Max Scaling
    print("\nData Hasil Normalisasi Min-Max Scaling:")
    normalized_data_minmax