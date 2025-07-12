import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler
import numpy as np
from streamlit_lottie import st_lottie
import json

# Memuat model dan scaler
model = joblib.load('best_model_XGB.pkl')
scaler = joblib.load('scaler_regresi.pkl')

# Memuat animasi Lottie
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Menambahkan gambar di bagian atas sidebar
# st.sidebar.image("logo-transparent.png", use_container_width=True)# Ganti dengan path gambar Anda


# Sidebar Navigasi Manual
st.sidebar.markdown("<h2 class='sidebar-header'>Navigasi</h2>", unsafe_allow_html=True)

# Menentukan halaman
pages = {
    "🏠 Dashboard": "Dashboard",
    "ℹ️ Indikator": "Indikator",
    "📊 Prediksi IKP": "Prediksi IKP"
}

# Mengingat halaman yang dipilih menggunakan session state
if 'selected_page' not in st.session_state:
    st.session_state.selected_page = "Dashboard"

# Tombol untuk memilih halaman tanpa menggunakan perulangan
home_button = st.sidebar.button("🏠 Dashboard", key="Dashboard")
about_button = st.sidebar.button("ℹ️ Indikator", key="Indikator")
content_button = st.sidebar.button("📊 Prediksi IKP", key="Prediksi IKP")

# Menangani navigasi tanpa perulangan
if home_button:
    st.session_state.selected_page = "Dashboard"
elif about_button:
    st.session_state.selected_page = "Indikator"
elif content_button:
    st.session_state.selected_page = "Prediksi IKP"

# Menampilkan halaman berdasarkan pemilihan
page = st.session_state.selected_page

# ---------------------- Dashboard ----------------------
if page == "Dashboard":
    st.markdown(
        """
        <div style="text-align:center">
            <h1 style="font-size: 60px;">KASI NAMA</h1>
            <h3>Belom Kasi nama </h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(""" 
    <div style='text-align: justify; font-size: 18px;'>
        <p>
           <strong>KasiNAMA</strong> kasi deskripsi.
    </div>
    """, unsafe_allow_html=True)

    #     # Embed iframe for the data visualization
    # st.markdown(""" 
    # <iframe src="https://data.goodstats.id/statistic/embed/provinsi-dengan-jumlah-penduduk-miskin-terbanyak-di-indonesia-maret-2023-qhMgC" frameborder="0" style="height: 380px; width: 100%"></iframe>
    # """, unsafe_allow_html=True)

# indikator
elif page == "Indikator":
    st.title("📊 Tentang Fitur-Fitur yang Digunakan")
    st.markdown("Berikut adalah fitur-fitur yang digunakan dalam sistem **Prediksi Ketahanan Pangan**:")

    # Dua kolom untuk membagi fitur
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(""" 
         **NCPR (X1) (Normative Consumption to Net Production Ratio)**  
        adalah perbandingan antara konsumsi pangan yang dibutuhkan atau dianggap normal dengan produksi pangan bersih suatu wilayah.

        **Kemiskinan (%)**  
        Persentase Kemiskinan Pada Suatu wilayah.

        **Tanpa Pangan (%)**  
        Persentase penduduk terhadap akses pangan pada suatu wilayah.

        **Tanpa Listrik (%)**  
        Persentase penduduk yang tidak memiliki akses ke listrik.

        **Tanpa Air bersih  (%)**  
        Persentase terhadap akses air bersih pada suaty wilayah.
        """)

    with col2:
        st.markdown(""" 
        **Lama Perempuan Bersekolah (Tahun)**  
        indikator mengenai lamanya perempuan bersekolah pada suatu wilayah.

        **Rasio Tenaga Kesehatan**  
        Rasio jumlah ketersediaan akses terhadap tenaga kesehatan.

        **Angka Harapan Hidup**  
        Angka harapan hidup bedasarkan umue.

        **Stunting (%)**  
        Presentase anak dengan stunting pada suatu wilayah.
                    
              """)

    st.markdown("---")
    st.success("Fitur-fitur di atas digunakan sebagai indikator  untuk menilai prediksi Ketahanan Pangan suatu wilayah.")

elif page == "Prediksi IKP":
    st.title("🧮 Prediksi Ketahanan Pangan")
    st.markdown("Masukkan data pada form di bawah ini untuk mengetahui Prediksi Ketahanan Pangan.")

    with st.form("input_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            feature1 = st.number_input('NCPR', min_value=0.0)
            feature2 = st.number_input('Kemiskinan', min_value=0.0)
            feature3 = st.number_input('Pengeluaran Pangan', min_value=0.0)
            feature4 = st.number_input('Tanpa Listrik ', min_value=0.0)
            feature5 = st.number_input('Tanpa air bersih', min_value=0.0)

        with col2:
            feature6 = st.number_input('Lama Sekolah untuk perempuan', min_value=0.0)
            feature7 = st.number_input('Rasio Tenaga Kesehatan', min_value=0.0)
            feature8 = st.number_input('Angka Harapan Hidup', min_value=0.0)
            feature9 = st.number_input('Stunting', min_value=0.0)
            

        submit_button = st.form_submit_button("🔍 Prediksi IKP")

    if submit_button:
        # Konversi input menjadi float untuk fitur 1-9 dan int untuk fitur ke-10
        input_data = pd.DataFrame([[ 
            float(feature1), float(feature2), float(feature3),
            float(feature4), float(feature5), float(feature6),
            float(feature7), float(feature8), float(feature9),
            
        ]])

        # Menyesuaikan dengan skala fitur yang telah dipelajari sebelumnya
        input_scaled = scaler.transform(input_data)

        # Prediksi dengan model yang telah dilatih
        prediction = model.predict(input_scaled)

        # Menampilkan hasil prediksi
        st.markdown("### 💡 Hasil Prediksi IKP:")
        st.write(f"**IKP yang diprediksi:** {prediction[0]:.2f}")
