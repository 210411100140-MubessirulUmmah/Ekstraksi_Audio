import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Ekstraksi Fitur",
    page_icon="🎶",
    layout="wide"
)

st.title("🎶 Ekstraksi Fitur")

st.write("File audio akan dicari statistiknya. Fitur-fitur yang diekstraksi termasuk statistik dari frekuensi, statistik Zero Crossing Rate (ZCR), dan RMS (Root Mean Square) energi signal dari setiap file audio.")


df = pd.read_csv("hasil_statistik2.csv")
st.write("Hasil Ekstraksi Fitur Ciri Audio : ")
st.write(df)
st.write("Penjelasan fitur-fitur yang ada")

st.write("""
    <ol>
        <li>Audio Mean : Rata-rata dari nilai amplitudo dalam sinyal audio. Ini menggambarkan tingkat kekuatan sinyal audio secara keseluruhan.</li>
        <li>Audio Median : Nilai mediannya dari amplitudo dalam sinyal audio. Median adalah nilai tengah dalam distribusi data.</li>
        <li>Audio Mode : Nilai yang paling sering muncul dalam sinyal audio.</li>
        <li>Audio Maxv : Nilai amplitudo maksimum dalam sinyal audio, menunjukkan puncak tertinggi dari sinyal tersebut.</li>
        <li>Audio Minv : Nilai amplitudo minimum dalam sinyal audio, menunjukkan puncak terendah dari sinyal tersebut. </li>
        <li>Audio Std : Deviasi standar dari amplitudo dalam sinyal audio, mengukur sejauh mana nilai-nilai amplitudo tersebar dari nilai rata-rata.</li>
        <li>Audio Skew : mengukur sejauh mana distribusi amplitudo dalam suara melengkung dari distribusi yang simetris. semakin padat sebaran (mengelompok) maka standar deviasinya dan variansinya rendah, semakin renggang sebaran (menyebar) maka standar deviasinya dan variansinya tinggi.</li>
        <li>Audio Kurtosis : Mengukur tingkat ketajaman puncak dalam distribusi amplitudo. semakin kecil nilai kurtosis maka grafik semakin landai. semakin besar nilai kurtosis maka grafik semakin meruncing</li>
        <li>Audio Q1 : (Kuartil Pertama) adalah nilai yang membagi bagian bawah 25% data terendah ketika data telah diurutkan.</li>
        <li>Audio Q3 : (Kuartil Ketiga) adalah nilai yang membagi bagian atas 25% data tertinggi ketika data telah diurutkan. Secara matematis, Q3 adalah median dari setengah bagian kedua dari data.</li>
        <li>Audio IQR : adalah selisih antara Q3 dan Q1.</li>
        <li>ZCR Mean : Rata-rata dari tingkat perubahan nol-crossing rate (ZCR) dalam sinyal audio. ZCR mengukur sejauh mana sinyal audio melintasi nilai nol.</li>
        <li>ZCR Median : Median dari tingkat perubahan nol-crossing rate (ZCR) dalam sinyal audio. ZCR mengukur sejauh mana sinyal audio melintasi nilai nol.</li>
        <li>ZCR std : Standar devisiasi dari tingkat perubahan nol-crossing rate (ZCR) dalam sinyal audio. ZCR mengukur sejauh mana sinyal audio melintasi nilai nol. </li>
        <li>ZCR Kurtosis : Kurtosis dari tingkat perubahan nol-crossing rate (ZCR) dalam sinyal audio. ZCR mengukur sejauh mana sinyal audio melintasi nilai nol.</li>
        <li>ZCR Skew : Skewness dari tingkat perubahan nol-crossing rate (ZCR) dalam sinyal audio. ZCR mengukur sejauh mana sinyal audio melintasi nilai nol.</li>
        <li>RMS Energi Mean : Rata-rata dari energi root mean square (RMS) dalam sinyal audio. RMS mengukur kekuatan rata-rata sinyal audio.</li>
        <li>RMS Energi Median : Median dari energi root mean square (RMS) dalam sinyal audio. RMS mengukur kekuatan rata-rata sinyal audio.</li>
        <li>RMS Energi std : Standar Devisiasi dari energi root mean square (RMS) dalam sinyal audio. RMS mengukur kekuatan rata-rata sinyal audio.</li>
        <li>RMS Energi kurtosis : Kurtosis dari energi root mean square (RMS) dalam sinyal audio. RMS mengukur kekuatan rata-rata sinyal audio.</li>
        <li>RMS Energi skew : Skewness dari energi root mean square (RMS) dalam sinyal audio. RMS mengukur kekuatan rata-rata sinyal audio.</li>
        <li>Label : Label atau kategori yang menunjukkan emosi atau klasifikasi lain dari sinyal audio, seperti marah, senang, sedih, dll.</li>
    </ol>
""",unsafe_allow_html=True)


