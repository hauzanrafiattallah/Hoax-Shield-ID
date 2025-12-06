import streamlit as st
import numpy as np
import joblib
from typing import Tuple
import time

APP_NAME = "HoaxShield ID"
PRIMARY_COLOR = "#10B981"
BACKGROUND_COLOR = "#0F172A"
COMPONENT_COLOR = "#1E293B"

st.set_page_config(
    page_title=f"{APP_NAME} – Analisis Berita Indonesia",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {BACKGROUND_COLOR};
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    
    h1, h2, h3 {{
        color: #F8FAFC !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em;
    }}
    
    p, li, div {{
        color: #94A3B8;
        line-height: 1.6;
    }}

    section[data-testid="stSidebar"] {{
        background-color: {COMPONENT_COLOR};
        border-right: 1px solid #334155;
    }}

    .stButton > button {{
        background-color: {PRIMARY_COLOR};
        color: #022C22;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        border-radius: 6px;
        transition: all 0.2s ease;
        width: 100%;
    }}

    .stButton > button:hover {{
        background-color: #34D399;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2);
    }}

    .css-card {{
        background-color: {COMPONENT_COLOR};
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }}

    .metric-container {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 6px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }}

    .status-label {{
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }}

    .result-value {{
        font-size: 1.5rem;
        font-weight: 700;
        color: #F8FAFC;
    }}

    .stTextArea textarea {{
        background-color: #020617;
        color: #F1F5F9;
        border: 1px solid #334155;
        border-radius: 6px;
    }}

    .stTextArea textarea:focus {{
        border-color: {PRIMARY_COLOR};
        box-shadow: 0 0 0 1px {PRIMARY_COLOR};
    }}

    .stProgress > div > div > div > div {{
        background-color: {PRIMARY_COLOR};
    }}
    
    hr {{
        border-color: #334155;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        model = joblib.load("news_classifier.pkl")
    except Exception as e:
        st.error("Sistem gagal memuat model klasifikasi. Pastikan file 'news_classifier.pkl' tersedia.")
        st.stop()
    return model

model = load_model()

def predict_text(text: str) -> Tuple[str, float]:
    if not text.strip():
        return "N/A", 0.0

    proba = model.predict_proba([text])[0]
    
    try:
        classes = list(model.classes_)
        idx_hoax = classes.index(1)
    except Exception:
        idx_hoax = 1

    proba_hoax = float(proba[idx_hoax])
    label = "Hoaks" if proba_hoax >= 0.5 else "Valid"
    return label, proba_hoax

st.sidebar.markdown(
    f"""
    <div style="padding: 1rem 0; text-align: left;">
        <h2 style="margin:0; font-size: 1.2rem; color: #10B981 !important;">🛡️ {APP_NAME}</h2>
        <p style="font-size: 0.8rem; margin-top: 5px; color: #64748B;">Sistem Deteksi Berita & Literasi Digital</p>
    </div>
    """, 
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "Menu Utama",
    (
        "Dashboard",
        "Analisis Berita",
        "Indikator Hoaks",
        "Metodologi",
        "Tentang Pengembang",
    ),
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style="font-size: 0.75rem; color: #64748B;">
    <strong>Panduan Singkat</strong><br>
    Pastikan teks berita yang dimasukkan menggunakan Bahasa Indonesia yang baik untuk hasil analisis yang optimal.
    </div>
    """,
    unsafe_allow_html=True
)

if page == "Dashboard":
    col_hero, col_info = st.columns([1.5, 1])

    with col_hero:
        st.markdown(f"<h1 style='margin-bottom: 1rem;'>Selamat Datang di {APP_NAME}</h1>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="css-card">
                <p style="margin-bottom: 0;">
                Platform ini dirancang untuk melakukan verifikasi awal terhadap validitas sebuah berita menggunakan pendekatan 
                <em>Natural Language Processing</em> (NLP). Sistem ini dilatih untuk mengenali pola linguistik yang umum ditemukan 
                pada berita palsu (hoaks) dibandingkan dengan jurnalisme standar.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown("### Fitur Utama")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                """
                <div class="css-card" style="padding: 1rem;">
                    <strong style="color: #F8FAFC;">Analisis Cepat</strong>
                    <p style="font-size: 0.9rem; margin-top: 0.5rem;">
                    Mendeteksi indikasi hoaks dalam hitungan detik menggunakan model Machine Learning.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c2:
            st.markdown(
                """
                <div class="css-card" style="padding: 1rem;">
                    <strong style="color: #F8FAFC;">Probabilitas Terukur</strong>
                    <p style="font-size: 0.9rem; margin-top: 0.5rem;">
                    Menampilkan skor keyakinan model terhadap prediksi yang diberikan.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

    with col_info:
        st.markdown(
            """
            <div class="css-card">
                <h4 style="color: #F8FAFC; margin-bottom: 1rem;">Spesifikasi Model</h4>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem; border-bottom: 1px solid #334155; padding-bottom: 0.5rem;">
                    <span>Algoritma</span>
                    <span style="color: #F8FAFC;">Logistic Regression</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem; border-bottom: 1px solid #334155; padding-bottom: 0.5rem;">
                    <span>Ekstraksi Fitur</span>
                    <span style="color: #F8FAFC;">TF-IDF Vectorizer</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem; border-bottom: 1px solid #334155; padding-bottom: 0.5rem;">
                    <span>Akurasi Test</span>
                    <span style="color: #10B981;">~98%</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span>Bahasa</span>
                    <span style="color: #F8FAFC;">Indonesia</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

elif page == "Analisis Berita":
    st.markdown("<h1>Analisis Validitas Berita</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        <p style='margin-bottom: 2rem;'>
        Silakan masukkan judul dan isi berita pada kolom di bawah ini. Sistem akan melakukan pemindaian pola teks.
        </p>
        """, 
        unsafe_allow_html=True
    )

    col_input, col_result = st.columns([1.3, 1])

    with col_input:
        user_input = st.text_area(
            "Input Teks Berita",
            height=300,
            placeholder="Tempelkan teks berita di sini..."
        )
        
        analyze_btn = st.button("Jalankan Analisis")

    with col_result:
        if analyze_btn:
            if not user_input.strip():
                st.error("Kesalahan: Input teks tidak boleh kosong. Harap masukkan berita yang valid.")
            else:
                with st.spinner("Sedang memproses algoritma prediksi..."):
                    time.sleep(0.5)
                    label, proba = predict_text(user_input)
                
                st.success("Proses analisis berhasil diselesaikan.")
                
                confidence_score = proba * 100
                is_hoax = label == "Hoaks"
                
                status_color = "#EF4444" if is_hoax else "#10B981"
                status_text = "TERINDIKASI HOAKS" if is_hoax else "TERINDIKASI VALID"
                
                st.markdown(
                    f"""
                    <div class="css-card" style="border-left: 4px solid {status_color};">
                        <div class="status-label" style="color: {status_color};">Hasil Klasifikasi</div>
                        <div class="result-value">{status_text}</div>
                        <p style="margin-top: 0.5rem; font-size: 0.9rem;">
                            Berdasarkan pola sintaksis dan semantik teks.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                st.markdown(
                    f"""
                    <div class="css-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                            <span style="font-size: 0.9rem;">Probabilitas Hoaks</span>
                            <span style="font-weight: bold; color: {status_color};">{confidence_score:.1f}%</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.progress(min(max(proba, 0.0), 1.0))
                
                interpretation = (
                    "Model mendeteksi pola bahasa yang sangat mirip dengan data latih kategori Hoaks. Disarankan untuk melakukan verifikasi silang."
                    if is_hoax else
                    "Struktur teks konsisten dengan pola berita standar (Valid). Namun, tetap pastikan sumber berita kredibel."
                )
                
                st.info(interpretation)
        
        else:
            st.markdown(
                """
                <div class="css-card" style="text-align: center; color: #64748B; padding: 3rem 1rem;">
                    <p>Menunggu input data untuk memulai analisis.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

elif page == "Indikator Hoaks":
    st.markdown("<h1>Indikator Umum Berita Hoaks</h1>", unsafe_allow_html=True)
    st.markdown("<p>Kenali ciri-ciri disinformasi untuk meningkatkan kewaspadaan digital.</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="css-card">
                <h3 style="color: #10B981 !important;">Judul Provokatif</h3>
                <p>Penggunaan huruf kapital berlebih, tanda seru ganda, dan pemilihan kata yang memancing emosi negatif (marah, takut) tanpa substansi yang jelas.</p>
            </div>
            <div class="css-card">
                <h3 style="color: #10B981 !important;">Sumber Anonim</h3>
                <p>Tidak mencantumkan nama penulis, redaksi, atau mengutip narasumber yang tidak dapat diverifikasi kredibilitasnya.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="css-card">
                <h3 style="color: #10B981 !important;">Distorsi Visual</h3>
                <p>Menggunakan foto atau video lama yang didaur ulang untuk konteks kejadian baru, atau hasil manipulasi penyuntingan digital.</p>
            </div>
            <div class="css-card">
                <h3 style="color: #10B981 !important;">Fanatisme & Bias</h3>
                <p>Nigrasi opini subjektif yang disajikan seolah-olah fakta objektif, seringkali menyerang kelompok tertentu secara tidak berimbang.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

elif page == "Metodologi":
    st.markdown("<h1>Metodologi & Sumber Data</h1>", unsafe_allow_html=True)
    
    st.markdown("### Dataset")
    st.markdown(
        """
        <div class="css-card">
            Sistem dikembangkan menggunakan dataset <strong>Deteksi Berita Hoaks Indo</strong> yang bersumber dari repositori publik. 
            Dataset mencakup ribuan artikel yang telah dikurasi dari berbagai sumber:
            <ul style="margin-top: 0.5rem; color: #94A3B8;">
                <li>TurnBackHoax.id (MAFINDO)</li>
                <li>Antara News</li>
                <li>Kompas</li>
                <li>Detik</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Arsitektur Sistem")
    st.markdown(
        """
        <div class="css-card">
            <ol style="margin-top: 0.5rem; color: #94A3B8; padding-left: 1.5rem;">
                <li style="margin-bottom: 0.5rem;"><strong>Preprocessing:</strong> Pembersihan teks, normalisasi karakter, dan tokenisasi.</li>
                <li style="margin-bottom: 0.5rem;"><strong>Feature Engineering:</strong> Menggunakan TF-IDF (Term Frequency-Inverse Document Frequency) untuk pembobotan kata.</li>
                <li style="margin-bottom: 0.5rem;"><strong>Modeling:</strong> Algoritma Logistic Regression dipilih karena efisiensi dan interpretabilitas yang baik pada data teks dimensi tinggi.</li>
                <li><strong>Evaluasi:</strong> Model diuji menggunakan metrik Akurasi, Presisi, Recall, dan AUC-ROC.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True
    )

elif page == "Tentang Pengembang":
    st.markdown("<h1>Informasi Pengembang</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="css-card">
            <p>
            Aplikasi ini merupakan hasil pengembangan riset di bidang <em>Artificial Intelligence</em>. 
            Tujuan utama proyek ini adalah menyediakan alat bantu (assistive tool) bagi masyarakat dalam menyaring informasi di era digital.
            </p>
            <p>
            <strong>Disclaimer:</strong> Hasil prediksi model adalah indikasi statistik berdasarkan pola data latih dan tidak bersifat absolut. 
            Pengguna diharapkan tetap melakukan verifikasi mandiri melalui kanal resmi.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )