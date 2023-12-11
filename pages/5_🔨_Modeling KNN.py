import streamlit as st

import numpy as np

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score

st.set_page_config(
    page_title="Modeling KNN",
    page_icon="🔨",
    layout="wide"
)

st.title("🔨 Modeling KNN")


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

# Normalisasi Min-Max pada data training
minmax_scaler = MinMaxScaler()
X_train_minmax = minmax_scaler.fit_transform(X_train)

# Membuat dictionary untuk menyimpan akurasi untuk setiap nilai k dari 1 hingga 100
accuracy_dict_zscore = {}
accuracy_dict_minmax = {}
data = []

# Melakukan loop untuk mencoba setiap nilai k dari 1 hingga 100
for k in range(1, 101):
    # Mendefinisikan dan melatih model KNN dengan Z-score
    knn_model_zscore = KNeighborsClassifier(n_neighbors=k)
    knn_model_zscore.fit(X_train_zscore, y_train)

    # Melakukan prediksi pada data testing yang telah dinormalisasi dengan Z-score
    X_test_zscore = zscore_scaler.transform(X_test)
    y_pred_zscore = knn_model_zscore.predict(X_test_zscore)

    # Mengukur akurasi model dengan Z-score
    accuracy_zscore = accuracy_score(y_test, y_pred_zscore)
    accuracy_dict_zscore[k] = accuracy_zscore

    # Mendefinisikan dan melatih model KNN dengan Min-Max Scaling
    knn_model_minmax = KNeighborsClassifier(n_neighbors=k)
    knn_model_minmax.fit(X_train_minmax, y_train)

    # Melakukan prediksi pada data testing yang telah dinormalisasi dengan Min-Max Scaling
    X_test_minmax = minmax_scaler.transform(X_test)
    y_pred_minmax = knn_model_minmax.predict(X_test_minmax)

    # Mengukur akurasi model dengan Min-Max Scaling
    accuracy_minmax = accuracy_score(y_test, y_pred_minmax)
    accuracy_dict_minmax[k] = accuracy_minmax

    result = {
        'Jumlah K' : k,
        'Akurasi Model KNN Z-Score' : accuracy_zscore,
        'Akurasi Model KNN MinMax' : accuracy_minmax
    }
    data.append(result)
hasil = pd.DataFrame(data)
hasilnoindex = hasil.to_csv('hasilnoindex.csv', index=False)

# Membuat grafik perbandingan akurasi Z-score dan Min-Max Scaling
plt.figure(figsize=(10, 6))
plt.plot(list(accuracy_dict_zscore.keys()), list(accuracy_dict_zscore.values()), label='Z-score')
plt.plot(list(accuracy_dict_minmax.keys()), list(accuracy_dict_minmax.values()), label='Min-Max')
plt.xlabel('k Value')
plt.ylabel('Accuracy')
plt.title('Perbandingan Akurasi nilai K menggunakan Z-score dan Min-Max Scaling')
plt.legend()
plt.grid(True)

# Menyimpan grafik ke dalam file gambar
plt.savefig('Perbandingan Akurasi MinMax dan Zscore.png')

# Menampilkan grafik menggunakan Streamlit
st.image('Perbandingan Akurasi MinMax dan Zscore.png', use_column_width=True)

# Menampilkan nilai k terbaik dan akurasinya untuk Z-score
best_k_zscore = max(accuracy_dict_zscore, key=accuracy_dict_zscore.get)
st.write(f"Best k value with Z-score : {best_k_zscore}, Accuracy : {accuracy_dict_zscore[best_k_zscore]*100:.2f}%")

# Menampilkan nilai k terbaik dan akurasinya untuk Min-Max Scaling
best_k_minmax = max(accuracy_dict_minmax, key=accuracy_dict_minmax.get)
st.write(f"Best k value with Min-Max : {best_k_minmax}, Accuracy : {accuracy_dict_minmax[best_k_minmax]*100:.2f}%\n\n\n")


# Memisahkan DataFrame menjadi dua bagian
baca = pd.read_csv('hasilnoindex.csv')
middle_index = len(hasil) // 2
left_half = baca.iloc[:middle_index]
right_half = baca.iloc[middle_index:]
# Tampilkan dua kolom
col1, col2 = st.columns(2)
# Tampilkan setengah tabel di kolom 1
col1.table(left_half)
# Tampilkan setengah tabel di kolom 2
col2.table(right_half)


best_k_zscore = max(accuracy_dict_zscore, key=accuracy_dict_zscore.get)  # Mendapatkan jumlah k terbaik dengan Z-score
best_k_minmax = max(accuracy_dict_minmax, key=accuracy_dict_minmax.get)  # Mendapatkan jumlah k terbaik dengan Min-Max Scaling

# Mendefinisikan dan melatih model KNN terbaik dengan Z-score
best_knn_model_zscore = KNeighborsClassifier(n_neighbors=best_k_zscore)
best_knn_model_zscore.fit(X_train_zscore, y_train)

# Mendefinisikan dan melatih model KNN terbaik dengan Min-Max Scaling
best_knn_model_minmax = KNeighborsClassifier(n_neighbors=best_k_minmax)
best_knn_model_minmax.fit(X_train_minmax, y_train)

# Simpan model terbaik ke dalam file menggunakan pickle
best_zscore_acc = accuracy_dict_zscore[best_k_zscore]*100
save_zscore = {
    'best_knn_model_zscore' : best_knn_model_zscore,
    'best_zscore_acc' : best_zscore_acc,
}
with open('best_knn_model_zscore.pkl', 'wb') as model_file:
    pickle.dump(save_zscore, model_file)

best_minmax_acc = accuracy_dict_minmax[best_k_minmax]*100
save_minmax = {
    'best_knn_model_minmax' : best_knn_model_minmax,
    'best_minmax_acc' : best_minmax_acc,
}
with open('best_knn_model_minmax.pkl', 'wb') as model_file:
    pickle.dump(save_minmax, model_file)