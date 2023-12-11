import streamlit as st

st.set_page_config(
    page_title="Dataset",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Dataset")


st.write("""<h3> Toronto emotional speech set (TESS) <h3>""", unsafe_allow_html=True)
st.write("""<h5> A dataset for training emotion (7 cardinal emotions) classification in audio </h5>""", unsafe_allow_html=True)

st.markdown("""
    <div style="text-align: justify">
        Dataset audio ini saya ambil dari kaggle yang dikumpulkan oleh University of Toronto dengan berkolaborasi bersama Eu Jin Lok sebagai author. Dataset ini di publis pada tahun 2019.\n Untuk bisa mengakses datasetnya bisa klik link :
        <a href="https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess?resource=download">Disini</a>
    </div>""", unsafe_allow_html=True)


st.write(
    """
    <div style="text-align: justify">
        Dataset ini berisi serangkaian 200 kata target yang diucapkan dalam kalimat "Katakan kata.." oleh dua 
        aktris perempuan (berusia 26 dan 64 tahun), dan rekaman dibuat dari setiap kata tersebut dengan tujuh emosi yang berbeda yakni 
        (marah, jijik, takut, bahagia, kejutan menyenangkan, sedih, dan netral). Total ada 2800 data (berupa file audio) 
        dalam format WAV.
        Maksud dari kalimat "Katakan kata .. ", adalah kalimat ini digunakan sebagai format standar di mana dua aktris 
        perempuan diminta untuk mengucapkan 200 kata target. Dengan demikian, kalimat ini bertindak sebagai 
        pola yang memandu cara kata-kata tersebut diucapkan dalam rekaman audio. Bagian ".." dalam kalimat tersebut menunjukkan tempat 
        di mana kata target akan dimasukkan. Misalnya, jika kata target adalah "apple", kalimat yang diucapkan akan menjadi "Katakan kata apple." 
        Ini memastikan konsistensi dalam pengucapan kata-kata target selama percobaan atau penelitian.
    </div>
    """, unsafe_allow_html=True)



image = open('Folder Data Audio.png', 'rb').read()
st.image(image, caption='Dataset Audio', use_column_width=True)