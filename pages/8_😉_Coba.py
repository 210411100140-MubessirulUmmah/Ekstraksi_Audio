import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

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
X_test_zscore = zscore_scaler.transform(X_test)

# Normalisasi Min-Max pada data training
minmax_scaler = MinMaxScaler()
X_train_minmax = minmax_scaler.fit_transform(X_train)
X_test_minmax = minmax_scaler.transform(X_test)

# Membuat dictionary untuk menyimpan akurasi untuk setiap nilai k dari 1 hingga 100

data1 = []
data2 = []

# Melakukan loop untuk mencoba setiap nilai k dari 1 hingga 100
k_values = [i for i in range (1,101)]
for k in k_values:
    # Mendefinisikan model KNN dengan Z-score
    knn_model_zscore = KNeighborsClassifier(n_neighbors=k)
    
    # Menghitung akurasi model dengan ShuffleSplit dan Z-score
    scores_zscore = cross_val_score(knn_model_zscore, X_train_zscore, y_train, cv=5)
    data1.append(np.mean(scores_zscore))

    # Mendefinisikan model KNN dengan Min-Max Scaling
    knn_model_minmax = KNeighborsClassifier(n_neighbors=k)

    
    # Menghitung akurasi model dengan ShuffleSplit dan Min-Max Scaling
    scores_minmax = cross_val_score(knn_model_minmax, X_train_minmax, y_train, cv=5)
    data2.append(np.mean(scores_minmax))

# sns.lineplot(x = k_values, y = data1, marker = 'o')
# plt.xlabel("K Values")
# plt.ylabel("Accuracy Score")
# plt.title('Accuracy Z-Score')

# sns.lineplot(x = k_values, y = data2, marker = 'o')
# plt.xlabel("K Values")
# plt.ylabel("Accuracy Score")
# plt.title('Accuracy MinMax')

# Memilih nilai k terbaik untuk Z-score
best_index_zscore = np.argmax(data1)
best_k_zscore = k_values[best_index_zscore]

knn_zscore = KNeighborsClassifier(n_neighbors=best_k_zscore)
knn_zscore.fit(X_train_zscore, y_train)

best_index_minmax = np.argmax(data2)
best_k_minmax = k_values[best_index_minmax]

knn_minmax = KNeighborsClassifier(n_neighbors=best_k_minmax)
knn_minmax.fit(X_train_minmax, y_train)


# Prediksi hasil pada data uji
y_pred_zscore = knn_zscore.predict(X_test_zscore)
y_pred_minmax = knn_minmax.predict(X_test_minmax)

# Menghitung akurasi prediksi pada data uji
accuracy_zscore_test = accuracy_score(y_test, y_pred_zscore)
accuracy_minmax_test = accuracy_score(y_test, y_pred_minmax)

# Menampilkan hasil akurasi prediksi pada data uji
st.write(f"Best k value with Z-score: {best_k_zscore}, Test Accuracy (Z-Score): {accuracy_zscore_test * 100:.2f}%")
st.write(f"Best k value with Min-Max: {best_k_minmax}, Test Accuracy (MinMax): {accuracy_minmax_test * 100:.2f}%")

save_zscore = {
    'best_knn_model_zscore' : knn_zscore,
    'best_zscore_acc' : accuracy_zscore_test*100,
    'scaler' : zscore_scaler
}
with open('cross_best_knn_model_zscore.pkl', 'wb') as model_file:
    pickle.dump(save_zscore, model_file)


save_minmax = {
    'best_knn_model_minmax' : knn_minmax,
    'best_minmax_acc' : accuracy_minmax_test*100,
    'scaler' : minmax_scaler
}
with open('cross_best_knn_model_minmax.pkl', 'wb') as model_file:
    pickle.dump(save_minmax, model_file)