import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score

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

df = pd.read_csv('hasil_statistik2.csv')

# Memisahkan kolom target (label) dari kolom fitur
X = df.drop(columns=['Label'])  # Kolom fitur
y = df['Label']  # Kolom target


# Membagi data menjadi data training dan data testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisasi Min-Max Scaling pada data
minmax_scaler = MinMaxScaler()
X_train_minmax = minmax_scaler.fit_transform(X_train)

X_test_minmax = minmax_scaler.transform(X_test)

Zscore_scaler = StandardScaler()
X_train_Zscore = Zscore_scaler.fit_transform(X_train)

X_test_Zscore = Zscore_scaler.transform(X_test)

accuracy_zscore = 0
n_zscore = 0
k_zscore = 0

accuracy_minmax = 0
n_minmax = 0
k_minmax = 0
data1 = []
data2 = []

for component in range(1, 21):
    accuracy_dict = {}
    for k_value in range(1, 101):
        pca_zscore = PCA(n_components=component)
        X_train_zscore_pca = pca_zscore.fit_transform(X_train_Zscore)
        X_test_zscore_pca = pca_zscore.transform(X_test_Zscore)

        knn_model_zscore = KNeighborsClassifier(n_neighbors=k_value)
        scores_zscore = cross_val_score(knn_model_zscore, X_train_zscore_pca, y_train, cv=5)
        data1.append(np.mean(scores_zscore))

        pca_minmax = PCA(n_components=component)
        X_train_minmax_pca = pca_minmax.fit_transform(X_train_minmax)
        X_test_minmax_pca = pca_minmax.transform(X_test_minmax)

        knn_model_minmax = KNeighborsClassifier(n_neighbors=k_value)
        scores_minmax = cross_val_score(knn_model_minmax, X_train_minmax_pca, y_train, cv=5)
        data1.append(np.mean(scores_minmax))

    if (np.mean(scores_zscore)) > accuracy_zscore:
        accuracy_zscore = (np.mean(scores_zscore))
        n_zscore = component
        k_zscore = k_value

    if (np.mean(scores_minmax)) > accuracy_minmax:
        accuracy_minmax = (np.mean(scores_minmax))
        n_minmax = component
        k_minmax = k_value

pcazscore = PCA(n_components=n_minmax)
X_train_minmax_pca = pcazscore.fit_transform(X_train_minmax)
X_test_minmax_pca = pcazscore.transform(X_test_minmax)
knn_model_minmax = KNeighborsClassifier(n_neighbors=k_minmax)
knn_model_minmax.fit(X_train_minmax_pca, y_train)

y_pred_minmax = knn_model_minmax.predict(X_test_minmax_pca)

accuracy_minmax_test = accuracy_score(y_test, y_pred_minmax)

pcaminmax = PCA(n_components=n_zscore)
X_train_zscore_pca = pcaminmax.fit_transform(X_train_Zscore)
X_test_zscore_pca = pcaminmax.transform(X_test_Zscore)
knn_model_zscore = KNeighborsClassifier(n_neighbors=k_zscore)
knn_model_zscore.fit(X_train_zscore_pca, y_train)

y_pred_zscore = knn_model_zscore.predict(X_test_zscore_pca)

accuracy_zscore_test = accuracy_score(y_test, y_pred_zscore)

# Menampilkan hasil akurasi prediksi pada data uji
st.write(f"Best k value with Z-score: {k_zscore}, Test Accuracy (Z-Score): {accuracy_zscore_test * 100:.2f}%")
st.write(f"Best k value with Min-Max: {k_minmax}, Test Accuracy (MinMax): {accuracy_minmax_test * 100:.2f}%")


save_zscore = {
    'best_knn_model_zscore' : knn_model_zscore,
    'best_zscore_acc' : accuracy_zscore_test*100,
    'scaler' : Zscore_scaler,
    'pca' : pcazscore
}
with open('cross_best_knn_model_pca_zscore.pkl', 'wb') as model_file:
    pickle.dump(save_zscore, model_file)


save_minmax = {
    'best_knn_model_minmax' : knn_model_minmax,
    'best_minmax_acc' : accuracy_minmax_test*100,
    'scaler' : minmax_scaler,
    'pca' : pcaminmax
}
with open('cross_best_knn_model_pca_minmax.pkl', 'wb') as model_file:
    pickle.dump(save_minmax, model_file)