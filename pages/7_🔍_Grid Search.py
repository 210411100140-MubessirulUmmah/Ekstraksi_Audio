import streamlit as st

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="Grid Search",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Grid Search")
    
st.markdown("<h4>Akurasi Model</h4>", unsafe_allow_html=True)
scaler = st.radio("", ('Grid Search Z-Score', 'Grid Search MinMax'))
if scaler == 'Grid Search Z-Score':
    st.title('Grid Search Metode KNN dengan Z-Score Scaler')
    # Membaca data dari file CSV
    df = pd.read_csv('hasil_statistik2.csv')

    # Memisahkan kolom target (label) dari kolom fitur
    X = df.drop(columns=['Label'])  # Kolom fitur
    y = df['Label']  # Kolom target

    # Memisahkan data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalisasi data menggunakan StandardScaler
    scaler = StandardScaler()
    X_train_zscore = scaler.fit_transform(X_train)
    X_test_zscore = scaler.transform(X_test)

    # Mendefinisikan parameter yang ingin diuji
    param_grid = {
        'n_neighbors': list(range(1, 101)),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    # Mendefinisikan model KNN
    knn = KNeighborsClassifier()

    # Mendefinisikan Grid Search dengan model KNN dan parameter yang diuji
    grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_zscore, y_train)  # Menggunakan data latih yang belum diproses PCA

    # Menampilkan parameter terbaik
    st.write("Best Parameters:", grid_search.best_params_)

    # Menggunakan PCA dengan komponen utama terbaik
    best_n_neighbors = grid_search.best_params_['n_neighbors']
    best_weights = grid_search.best_params_['weights']
    best_metric = grid_search.best_params_['metric']

    accuracy_dict = {}
    for n_components in range(X_train.shape[1], 0, -1):
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train_zscore)
        X_test_pca = pca.transform(X_test_zscore)

        # Membuat model KNN dengan hyperparameter terbaik
        best_knn_model = KNeighborsClassifier(n_neighbors=best_n_neighbors, weights=best_weights, metric=best_metric)
        best_knn_model.fit(X_train_pca, y_train)

        # Membuat prediksi menggunakan model terbaik
        y_pred = best_knn_model.predict(X_test_pca)

        # Mengukur akurasi model terbaik pada data uji
        grid_knn_pca = accuracy_score(y_test, y_pred)

        # Menyimpan akurasi dalam dictionary
        accuracy_dict[n_components] = grid_knn_pca

        st.write(f"Accuracy dengan {n_components} PCA components: {grid_knn_pca * 100:.2f}%")

    # Mencari nilai k terbaik
    best_comp = max(accuracy_dict, key=accuracy_dict.get)
    best_accuracy = accuracy_dict[best_comp] * 100
    st.write(f"\nBest Accuracy pada Grid Search KNN Z-Score Scaler {best_comp} PCA components: {best_accuracy:.2f}%")
    
    pca = PCA(n_components=best_comp)
    X_train_pca = pca.fit_transform(X_train_zscore)
    X_test_pca = pca.transform(X_test_zscore)
    knn_model = KNeighborsClassifier(n_neighbors=best_n_neighbors, weights=best_weights, metric=best_metric)
    knn_model.fit(X_train_pca, y_train)

    # Simpan akurasi ke dalam file menggunakan pickle
    best_zscore_grid_acc = best_accuracy
    save_zscore_reduksi = {
        'best_knn_model_zscore_pca' : knn_model,
        'best_zscore_grid_acc' : best_zscore_grid_acc,
        'pca' : pca,
    }
    with open('best_knn_model_zscore_grid.pkl', 'wb') as scaler_file:
        pickle.dump(save_zscore_reduksi, scaler_file)

    # Menampilkan grafik komponen PCA vs. Akurasi
    components = list(accuracy_dict.keys())
    accuracies = list(accuracy_dict.values())

    plt.figure(figsize=(10, 6))
    plt.plot(components, accuracies, marker='o', color='r', label='Akurasi')
    plt.xlabel('Jumlah Komponen PCA')
    plt.ylabel('Akurasi')
    plt.title('Akurasi vs. Jumlah Komponen PCA')
    plt.xticks(np.arange(min(components), max(components)+1, 1))
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)

elif scaler == "Grid Search MinMax":
    st.title('Grid Search Metode KNN dengan MinMax Scaler')
    # Membaca data dari file CSV
    df = pd.read_csv('hasil_statistik2.csv')

    # Memisahkan kolom target (label) dari kolom fitur
    X = df.drop(columns=['Label'])  # Kolom fitur
    y = df['Label']  # Kolom target

    # Memisahkan data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalisasi data menggunakan StandardScaler
    scaler = MinMaxScaler()
    X_train_minmax = scaler.fit_transform(X_train)
    X_test_minmax = scaler.transform(X_test)

    # Mendefinisikan parameter yang ingin diuji
    param_grid = {
        'n_neighbors': list(range(1, 101)),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    # Mendefinisikan model KNN
    knn = KNeighborsClassifier()

    # Mendefinisikan Grid Search dengan model KNN dan parameter yang diuji
    grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_minmax, y_train)  # Menggunakan data latih yang belum diproses PCA

    # Menampilkan parameter terbaik
    st.write("Best Parameters:", grid_search.best_params_)

    # Menggunakan PCA dengan komponen utama terbaik
    best_n_neighbors = grid_search.best_params_['n_neighbors']
    best_weights = grid_search.best_params_['weights']
    best_metric = grid_search.best_params_['metric']

    accuracy_dict = {}
    for n_components in range(X_train_minmax.shape[1], 0, -1):
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train_minmax)
        X_test_pca = pca.transform(X_test_minmax)

        # Membuat model KNN dengan hyperparameter terbaik
        best_knn_model = KNeighborsClassifier(n_neighbors=best_n_neighbors, weights=best_weights, metric=best_metric)
        best_knn_model.fit(X_train_pca, y_train)

        # Membuat prediksi menggunakan model terbaik
        y_pred = best_knn_model.predict(X_test_pca)

        # Mengukur akurasi model terbaik pada data uji
        grid_knn_pca = accuracy_score(y_test, y_pred)

        # Menyimpan akurasi dalam dictionary
        accuracy_dict[n_components] = grid_knn_pca

        st.write(f"Accuracy dengan {n_components} PCA components : {grid_knn_pca * 100:.2f}%")

    # Mencari nilai k terbaik
    best_comp = max(accuracy_dict, key=accuracy_dict.get)
    best_accuracy = accuracy_dict[best_comp] * 100
    st.write(f"\nBest Accuracy pada Grid Search KNN MinMax Scaler {best_comp} PCA components: {best_accuracy:.2f}%")
    
    pca = PCA(n_components=best_comp)
    X_train_pca = pca.fit_transform(X_train_minmax)
    X_test_pca = pca.transform(X_test_minmax)
    knn_model = KNeighborsClassifier(n_neighbors=best_n_neighbors, weights=best_weights, metric=best_metric)
    knn_model.fit(X_train_pca, y_train)

    # Simpan akurasi ke dalam file menggunakan pickle
    best_minmax_grid_acc = best_accuracy
    save_minmax_reduksi = {
        'best_knn_model_minmax_pca' : knn_model,
        'best_minmax_grid_acc' : best_minmax_grid_acc,
        'pca' : pca,
    }
    with open('best_knn_model_minmax_grid.pkl', 'wb') as scaler_file:
        pickle.dump(save_minmax_reduksi, scaler_file)

    # Menampilkan grafik komponen PCA vs. Akurasi
    components = list(accuracy_dict.keys())
    accuracies = list(accuracy_dict.values())

    plt.figure(figsize=(10, 6))
    plt.plot(components, accuracies, marker='o', color='r', label='Akurasi')
    plt.xlabel('Jumlah Komponen PCA')
    plt.ylabel('Akurasi')
    plt.title('Akurasi vs. Jumlah Komponen PCA')
    plt.xticks(np.arange(min(components), max(components)+1, 1))
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)