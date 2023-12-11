import streamlit as st
from audioop import minmax

import matplotlib.pyplot as plt


import IPython
import numpy as np
import scipy.stats
import seaborn as sns
import pandas as pd
import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import streamlit as st
import librosa
from scipy.stats import skew, kurtosis, mode, iqr

st.set_page_config(
    page_title="Extraksi Data Audio",
    page_icon="🚀",
    layout="wide"
)


st.write("""
<center><h3><b>Nama : Mubessirul Ummah\n
NIM : 210411100140</b></h3></center>
""", unsafe_allow_html=True)

st.title("🤓 Prediksi Class")


st.markdown("<h4>Perbandingan Akurasi antara Metode KNN</h4>", unsafe_allow_html=True)

st.image('Perbandingan Akurasi MinMax dan Zscore.png')


def calculate_statistics(audio_path):
    x, sr = librosa.load(audio_path)

    mean = np.mean(x)
    std = np.std(x)
    maxv = np.amax(x)
    minv = np.amin(x)
    median = np.median(x)
    skewness = skew(x)
    kurt = kurtosis(x)
    q1 = np.quantile(x, 0.25)
    q3 = np.quantile(x, 0.75)
    mode_v = mode(x)[0]
    iqr = q3 - q1

    zcr = librosa.feature.zero_crossing_rate(x)
    mean_zcr = np.mean(zcr)
    median_zcr = np.median(zcr)
    std_zcr = np.std(zcr)
    kurtosis_zcr = kurtosis(zcr, axis=None)
    skew_zcr = skew(zcr, axis=None)

    n = len(x)
    mean_rms = np.sqrt(np.mean(x**2) / n)
    median_rms = np.sqrt(np.median(x**2) / n)
    skew_rms = np.sqrt(skew(x**2) / n)
    kurtosis_rms = np.sqrt(kurtosis(x**2) / n)
    std_rms = np.sqrt(np.std(x**2) / n)

    return [mean, median, mode_v, maxv, minv, std, skewness, kurt, q1, q3, iqr, mean_zcr, median_zcr, std_zcr, kurtosis_zcr, skew_zcr, mean_rms, median_rms, std_rms, kurtosis_rms, skew_rms]

uploaded_file = st.file_uploader("Pilih file audio...", type=["wav","mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    st.title("Prediksi Class Data Audio :")

    if st.button("Cek Nilai Statistik"):
        # Simpan file audio yang diunggah
        audio_path = "audio_diunggah.wav"
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Hitung statistik untuk file audio yang diunggah
        statistik = calculate_statistics(audio_path)

        results = []
        result = {
            'Audio Mean': statistik[0],
            'Audio Median': statistik[1],
            'Audio Mode': statistik[2],
            'Audio Maxv': statistik[3],
            'Audio Minv': statistik[4],
            'Audio Std': statistik[5],
            'Audio Skew': statistik[6],
            'Audio Kurtosis': statistik[7],
            'Audio Q1': statistik[8],
            'Audio Q3': statistik[9],
            'Audio IQR': statistik[10],
            'ZCR Mean': statistik[11],
            'ZCR Median': statistik[12],
            'ZCR Std': statistik[13],
            'ZCR Kurtosis': statistik[14],
            'ZCR Skew': statistik[15],
            'RMS Energi Mean': statistik[16],
            'RMS Energi Median': statistik[17],
            'RMS Energi Std': statistik[18],
            'RMS Energi Kurtosis': statistik[19],
            'RMS Energi Skew': statistik[20],
        }
        results.append(result)
        df = pd.DataFrame(results)
        st.write(df)


    if st.button("Deteksi Audio"):

        # Memuat data audio yang diunggah dan menyimpannya sebagai file audio
        audio_path = "audio_diunggah.wav"
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Menghitung statistik untuk file audio yang diunggah (gunakan fungsi calculate_statistics sesuai kebutuhan)
        audio_features = calculate_statistics(audio_path)
        results = []
        result = {
            'Audio Mean': audio_features[0],
            'Audio Median': audio_features[1],
            'Audio Mode': audio_features[2],
            'Audio Maxv': audio_features[3],
            'Audio Minv': audio_features[4],
            'Audio Std': audio_features[5],
            'Audio Skew': audio_features[6],
            'Audio Kurtosis': audio_features[7],
            'Audio Q1': audio_features[8],
            'Audio Q3': audio_features[9],
            'Audio IQR': audio_features[10],
            'ZCR Mean': audio_features[11],
            'ZCR Median': audio_features[12],
            'ZCR Std': audio_features[13],
            'ZCR Kurtosis': audio_features[14],
            'ZCR Skew': audio_features[15],
            'RMS Energi Mean': audio_features[16],
            'RMS Energi Median': audio_features[17],
            'RMS Energi Std': audio_features[18],
            'RMS Energi Kurtosis': audio_features[19],
            'RMS Energi Skew': audio_features[20],
        }
        results.append(result)
        data_implementasi = pd.DataFrame(results)

        
        with open('cross_best_knn_model_zscore.pkl', 'rb') as model_file:
            saved_data1 = pickle.load(model_file)
        modelzscore = saved_data1['best_knn_model_zscore']
        scalerzscore = saved_data1['scaler']
        with open('cross_best_knn_model_minmax.pkl', 'rb') as model_file:
            saved_data2 = pickle.load(model_file)
        modelminmax = saved_data2['best_knn_model_minmax']
        scalerminmax = saved_data1['scaler']

        # with open('cross_best_knn_model_pca_zscore.pkl', 'rb') as model_file:
        #     saved_data3 = pickle.load(model_file)
        # modelzscorepca = saved_data3['best_knn_model_zscore']
        # zscorepca = saved_data3['pca']
        # zscore_scaler = saved_data3['scaler']

        # with open('cross_best_knn_model_pca_minmax.pkl', 'rb') as model_file:
        #     saved_data4 = pickle.load(model_file)
        # modelminmaxpca = saved_data4['best_knn_model_minmax']
        # minmaxpca = saved_data4['pca']
        # minmax_scaler = saved_data4['scaler']

        # with open('best_knn_model_zscore_grid.pkl', 'rb') as model_file:
        #     saved_data5 = pickle.load(model_file)
        # modelzscoregrid = saved_data5['best_knn_model_zscore_pca']
        # pcazscoregrid = saved_data5['pca']
        # with open('best_knn_model_minmax_grid.pkl', 'rb') as model_file:
        #     saved_data6 = pickle.load(model_file)
        # modelminmaxgrid = saved_data6['best_knn_model_minmax_pca']
        # pcaminmaxgrid = saved_data6['pca']

        X_test_zscore = scalerzscore.transform(data_implementasi)

        X_test_minmax = scalerminmax.transform(data_implementasi)

        predict_label1 = modelzscore.predict(X_test_zscore)
        predict_label2 = modelminmax.predict(X_test_minmax)

        # implementasi_zscore_transform = zscore_scaler.transform(data_implementasi)
        # implementasi_zscorepca = zscorepca.transform(implementasi_zscore_transform)
        # predict_label3 = modelzscorepca.predict(implementasi_zscorepca)

        # implementasi_minmax_transform = minmax_scaler.transform(data_implementasi)
        # implementasi_minmaxpca = minmaxpca.transform(implementasi_minmax_transform)
        # predict_label4 = modelminmaxpca.predict(implementasi_minmaxpca)

        # implementasi_zscoregrid = pcazscoregrid.transform(data_implementasi)
        # predict_label5 = modelzscoregrid.predict(implementasi_zscoregrid)

        # implementasi_minmaxgrid = pcaminmaxgrid.transform(data_implementasi)
        # predict_label6 = modelminmaxgrid.predict(implementasi_minmaxgrid)
        


        # Menampilkan hasil prediksi
        tampil = []
        hasil = {
            'Prediksi Class Z-Score' : predict_label1,
            'Prediksi Class MinMax' : predict_label2,
            # 'Prediksi Class Z-Score PCA' : predict_label3,
            # 'Prediksi Class MinMax PCA' : predict_label4
            # 'Prediksi Class Z-Score Grid Search' : predict_label5,
            # 'Prediksi Class MinMax Grid Search' : predict_label6
        }
        tampil.append(hasil)
        df = pd.DataFrame(tampil)
        df_melted = pd.melt(df, var_name='Metode Prediksi', value_name='Hasil Prediksi')
        st.write(df_melted)

