import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import streamlit as st

st.set_page_config(
    page_title="Reduksi PCA",
    page_icon="✨",
    layout="wide"
)
st.title("✨ Reduksi PCA")

st.markdown("<h4>Akurasi Model</h4>", unsafe_allow_html=True)
scaler = st.radio("", ('ReduksiZscore', 'ReduksiMinMax'))
if scaler == 'ReduksiMinMax':
    # Membaca data dari file CSV
    df = pd.read_csv('hasil_statistik2.csv')

    # Memisahkan kolom target (label) dari kolom fitur
    X = df.drop(columns=['Label'])  # Kolom fitur
    y = df['Label']  # Kolom target

    # Streamlit UI dengan slider untuk memilih nilai n_components dan K
    st.title('K-Nearest Neighbors Classifier with Min-Max Scaling and PCA')
    n_components = st.slider("Choose the number of PCA components :", min_value=1, max_value=20, value=16, step=1)
    k_value = st.slider("Choose the value of K :", min_value=1, max_value=100, value=9, step=1)


    # Membagi data menjadi data training dan data testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalisasi Min-Max Scaling pada data
    minmax_scaler = MinMaxScaler()
    X_train_minmax = minmax_scaler.fit_transform(X_train)

    X_test_minmax = minmax_scaler.transform(X_test)
    # Melakukan PCA dengan nilai n_components yang dipilih
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_minmax)
    X_test_pca = pca.transform(X_test_minmax)

    # Mendefinisikan dan melatih model KNN dengan nilai k yang dipilih
    knn_model = KNeighborsClassifier(n_neighbors=k_value)
    knn_model.fit(X_train_pca, y_train)

    # Melakukan prediksi pada data testing yang telah direduksi oleh PCA
    y_pred = knn_model.predict(X_test_pca)

    # Mengukur akurasi model
    accuracy = accuracy_score(y_test, y_pred)

    # Menampilkan hasil akurasi
    st.write(f"Akurasi Model dengan n_components = {n_components} dan k = {k_value}: {accuracy:.4f}")

    # Menampilkan nilai k terbaik (k dengan akurasi tertinggi)
    st.write("Tabel Nilai k Terbaik untuk Setiap Komponen PCA :")
    best_k_values = []
    best_accuracies = []
    accuracy = 0
    n = 0
    k = 0

    for component in range(1, 21):
        accuracy_dict = {}
        for k in range(1, 101):
            knn_model = KNeighborsClassifier(n_neighbors=k)
            knn_model.fit(X_train_pca[:, :component], y_train)
            y_pred = knn_model.predict(X_test_pca[:, :component])
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_dict[k] = accuracy

        best_k = max(accuracy_dict, key=accuracy_dict.get)
        best_accuracy = accuracy_dict[best_k]
        best_k_values.append(best_k)
        best_accuracies.append(best_accuracy)

        if best_accuracy > accuracy:
            accuracy = best_accuracy
            n = n_components
            k = best_k

    # Menampilkan tabel nilai k terbaik dan akurasi untuk setiap komponen PCA
    data = {'Component': list(range(1, 21)), 'Best K': best_k_values, 'Accuracy': best_accuracies}
    df_result = pd.DataFrame(data)
    st.write(df_result)
    st.write(f"Nilai akurasi reduksi pca pada knn dengan minmax scaler yakni {accuracy*100:.2f}% dengan k {k} pada komponen {n}")

    pca = PCA(n_components=n)
    X_train_pca = pca.fit_transform(X_train_minmax)
    X_test_pca = pca.transform(X_test_minmax)
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train_pca, y_train)

    # Simpan akurasi ke dalam file menggunakan pickle
    best_minmax_pca_acc = accuracy
    save_minmax_reduksi = {
        'best_knn_model_minmax_pca' : knn_model,
        'best_minmax_pca_acc' : best_minmax_pca_acc,
        'pca' : pca,
    }
    with open('best_knn_model_minmax_pca.pkl', 'wb') as scaler_file:
        pickle.dump(save_minmax_reduksi, scaler_file)

elif scaler == "ReduksiZscore":
    # Membaca data dari file CSV
    df = pd.read_csv('hasil_statistik2.csv')

    # Memisahkan kolom target (label) dari kolom fitur
    X = df.drop(columns=['Label'])  # Kolom fitur
    y = df['Label']  # Kolom target

    # Streamlit UI dengan slider untuk memilih nilai n_components dan K
    st.title('K-Nearest Neighbors Classifier with Z-Score Scaling and PCA')
    n_components = st.slider("Choose the number of PCA components :", min_value=1, max_value=20, value=14, step=1)
    k_value = st.slider("Choose the value of K :", min_value=1, max_value=100, value=3, step=1)


    # Membagi data menjadi data training dan data testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalisasi Min-Max Scaling pada data
    Zscore_scaler = StandardScaler()
    X_train_Zscore = Zscore_scaler.fit_transform(X_train)

    X_test_Zscore = Zscore_scaler.transform(X_test)
    # Melakukan PCA dengan nilai n_components yang dipilih
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_Zscore)
    X_test_pca = pca.transform(X_test_Zscore)

    # Mendefinisikan dan melatih model KNN dengan nilai k yang dipilih
    knn_model = KNeighborsClassifier(n_neighbors=k_value)
    knn_model.fit(X_train_pca, y_train)

    # Melakukan prediksi pada data testing yang telah direduksi oleh PCA
    y_pred = knn_model.predict(X_test_pca)

    # Mengukur akurasi model
    accuracy = accuracy_score(y_test, y_pred)

    # Menampilkan hasil akurasi
    st.write(f"Akurasi Model dengan n_components = {n_components} dan k = {k_value}: {accuracy:.4f}")

    # Menampilkan nilai k terbaik (k dengan akurasi tertinggi)
    st.write("Tabel Nilai k Terbaik untuk Setiap Komponen PCA :")
    best_k_values = []
    best_accuracies = []
    accuracy = 0
    n = 0
    k = 0

    for component in range(1, 21):
        accuracy_dict = {}
        for k in range(1, 101):
            knn_model = KNeighborsClassifier(n_neighbors=k)
            knn_model.fit(X_train_pca[:, :component], y_train)
            y_pred = knn_model.predict(X_test_pca[:, :component])
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_dict[k] = accuracy

        best_k = max(accuracy_dict, key=accuracy_dict.get)
        best_accuracy = accuracy_dict[best_k]
        best_k_values.append(best_k)
        best_accuracies.append(best_accuracy)

        if best_accuracy > accuracy:
            accuracy = best_accuracy
            n = n_components
            k = best_k

    # Menampilkan tabel nilai k terbaik dan akurasi untuk setiap komponen PCA
    data = {'Component': list(range(1, 21)), 'Best K': best_k_values, 'Accuracy': best_accuracies}
    df_result = pd.DataFrame(data)
    st.write(df_result)
    st.write(f"Nilai akurasi reduksi pca pada knn dengan zscore scaler yakni {accuracy*100:.2f}% dengan k {k} pada komponen {n}")

    pca = PCA(n_components=n)
    X_train_pca = pca.fit_transform(X_train_Zscore)
    X_test_pca = pca.transform(X_test_Zscore)
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train_pca, y_train)

    # Simpan akurasi ke dalam file menggunakan pickle
    best_zscore_pca_acc = accuracy
    save_zscore_reduksi = {
        'best_knn_model_zscore_pca' : knn_model,
        'best_zscore_pca_acc' : best_zscore_pca_acc,
        'pca' : pca,
    }
    with open('best_knn_model_zscore_pca.pkl', 'wb') as scaler_file:
        pickle.dump(save_zscore_reduksi, scaler_file)
