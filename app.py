import streamlit as st
import numpy as np
import joblib
from typing import Tuple


# CONFIG & STYLE


APP_NAME = "HoaxShield ID"
PRIMARY_COLOR = "#39D177"

st.set_page_config(
    page_title=f"{APP_NAME} – Deteksi Berita Hoaks Indonesia",
    page_icon="🛡️",
    layout="wide",
)

# Custom CSS 
st.markdown(
    f"""
    <style>

    .main {{
        color: inherit;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
        background: transparent;
    }}


    section[data-testid="stSidebar"] {{
      background: transparent !important;
      border-right: 1px solid var(--text-color);
      opacity: .20;
      }}

    
    h1, h2, h3, h4 {{
        font-weight: 700 !important;
        color: inherit !important;
    }}

    
    .card {{
        border-radius: 18px;
        padding: 1.5rem 1.75rem;
        border: 1px solid rgba(0,0,0,0.1);
        background: transparent !important;
        box-shadow: none;
    }}

    .pill {{
        display: inline-flex;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.75rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        border: 1px solid rgba(148,163,184,0.35);
        color: inherit;
        background: transparent;
    }}

    .accent-pill {{
        border-color: {PRIMARY_COLOR};
        color: {PRIMARY_COLOR};
        background: rgba(57, 209, 119, 0.15);
    }}

    .metric-label {{
        font-size: 0.8rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: inherit;
        opacity: .75;
    }}

    .metric-value {{
        font-size: 2.1rem;
        font-weight: 700;
    }}

    .label-real {{
        color: #22C55E;
        background: rgba(34,197,94,0.12);
        border-radius: 999px;
        padding: 0.35rem 0.9rem;
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
    }}

    .label-hoax {{
        color: #F97373;
        background: rgba(239,68,68,0.12);
        border-radius: 999px;
        padding: 0.35rem 0.9rem;
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
    }}

    .label-dot {{
        width: 8px;
        height: 8px;
        border-radius: 999px;
        background: currentColor;
    }}

    .footer-text {{
        font-size: 0.78rem;
        opacity: .7;
        margin-top: 2rem;
    }}

    
    .stTextArea textarea {{
        border-radius: 16px;
        border: 1px solid rgba(0,0,0,0.2);
        background: transparent !important;
        color: inherit !important;
    }}

    
    .stButton > button {{
        background: {PRIMARY_COLOR};
        color: black;
        border-radius: 999px;
        border: none;
        font-weight: 600;
        padding: 0.6rem 1.4rem;
    }}

    .stButton > button:hover {{
        filter: brightness(1.08);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)



# LOAD MODEL


@st.cache_resource(show_spinner=True)
def load_model():
    """
    Load trained classifier from news_classifier.pkl.

    Assumes that the pickle contains a scikit-learn Pipeline
    with a vectorizer + classifier that supports:
        - model.predict([text])
        - model.predict_proba([text])
    and that class '1' corresponds to HOAX.
    """
    try:
        model = joblib.load("news_classifier.pkl")
    except Exception as e:
        st.error(
            "Gagal memuat file `news_classifier.pkl`. "
            "Pastikan file tersebut berada di direktori yang sama dengan `app.py`."
        )
        st.exception(e)
        st.stop()
    return model


model = load_model()


def predict_text(text: str) -> Tuple[str, float]:
    """
    Return predicted label ('Real' or 'Hoaks') and hoax probability (0–1).
    """
    if not text.strip():
        return "N/A", 0.0

    # Predict probabilities
    proba = model.predict_proba([text])[0]

    # safer: cari index class=1 (Hoax)
    try:
        classes = list(model.classes_)
        idx_hoax = classes.index(1)
    except Exception:
        # fallback: asumsi kelas [0, 1]
        idx_hoax = 1

    proba_hoax = float(proba[idx_hoax])

    label = "Hoaks" if proba_hoax >= 0.5 else "Real"
    return label, proba_hoax



# SIDEBAR NAVIGATION


st.sidebar.markdown(
    f"""
    <div style="padding-top:0.5rem; padding-bottom:0.75rem;">
        <span class="pill accent-pill">🛡️ {APP_NAME}</span>
        <div style="margin-top:0.75rem; font-size:0.8rem;">
            Deteksi berita hoaks berbahasa Indonesia berbasis NLP
            (TF-IDF + Logistic Regression).
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigasi",
    (
        "Beranda",
        "Deteksi Hoaks",
        "Ciri-ciri Berita Hoaks",
        "Sumber Data & Metode",
        "Tentang Tim",
    ),
)

st.sidebar.markdown(
    """
    ---
    **Tips:**  
    • Masukkan teks berita selengkap mungkin.  
    • Gunakan bahasa Indonesia.  
    """
)
st.sidebar.markdown(
    '<div style="font-size:0.75rem; color:#FFFFF;">💡 Untuk keperluan riset & edukasi, bukan untuk kepentingan hukum atau komersial.</div>',
    unsafe_allow_html=True,
)

# PAGES

if page == "Beranda":
    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.markdown('<span class="pill">Hoax Detection • NLP</span>', unsafe_allow_html=True)
        st.markdown(
            f"<h1 style='margin-top:0.6rem;'>{APP_NAME}</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <p style="font-size:1.02rem;">
            Platform sederhana untuk <b>mengecek apakah sebuah berita cenderung hoaks atau real</b> 
            menggunakan model machine learning yang dilatih pada ribuan berita online Indonesia
            (TurnBackHoax, Antara, Kompas, Detik).
            </p>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="card" style="margin-top:1rem;">
                <div class="metric-label">Kenapa aplikasi ini dibuat?</div>
                <p style="font-size:0.94rem; margin-top:0.4rem;">
                    Penyebaran hoaks di media sosial dan aplikasi chat semakin masif. 
                    {APP_NAME} membantu pengguna memiliki <b>second opinion</b> saat membaca berita, 
                    dengan memberikan indikasi probabilitas hoaks berdasarkan pola bahasa.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("")
        st.markdown("### Cara pakai singkat")
        st.markdown(
            """
            1. Buka menu **Deteksi Hoaks** di sidebar.  
            2. Paste teks berita (judul + isi) ke dalam kotak yang tersedia.  
            3. Klik **Cek Berita** dan lihat hasil prediksi + probabilitas hoaks.  
            4. Baca juga halaman **Ciri-ciri Berita Hoaks** untuk edukasi tambahan.
            """
        )

    with col2:
        st.markdown(
            f"""
            <div class="card" style="margin-top:0.5rem;">
                <div class="metric-label">Model</div>
                <div style="margin-top:0.35rem; font-size:0.95rem;">
                    <b>TF-IDF + Logistic Regression</b><br/>
                    <span style="font-size:0.8rem;">
                        Dilatih dengan dataset <b>Deteksi Berita Hoaks Indo</b> (Kaggle).
                    </span>
                </div>
                <div style="margin-top:1.1rem;">
                    <div class="metric-label">Performa (Test Set)</div>
                    <div class="metric-value">≈ 0.98</div>
                    <div style="font-size:0.8rem;">Accuracy</div>
                </div>
                <hr style="border-color:#111827; margin:1rem 0 0.9rem 0;"/>
                <ul style="font-size:0.85rem; padding-left:1.2rem;">
                    <li>AUC ROC ≈ 0.998</li>
                    <li>Bahasa: Indonesia</li>
                    <li>Jenis tugas: biner (Real vs Hoaks)</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="footer-text">
        ⚠️ <b>Disclaimer:</b> {APP_NAME} tidak menggantikan verifikasi fakta profesional.
        Gunakan sebagai alat bantu, bukan satu-satunya sumber kebenaran.
        </div>
        """,
        unsafe_allow_html=True,
    )

elif page == "Deteksi Hoaks":
    st.markdown('<span class="pill accent-pill">📰 Deteksi Berita</span>', unsafe_allow_html=True)
    st.markdown("<h1>Cek Berita Hoaks atau Real</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        Tempelkan teks berita (judul + isi singkat) pada kotak di bawah.  
        Model akan mengembalikan prediksi apakah berita cenderung **Real** atau **Hoaks**, 
        beserta skor probabilitas hoaks.
        """
    )

    col_input, col_result = st.columns([1.4, 1])

    with col_input:
        default_text = (
            "Jakarta (ANTARA) – Anggota DPR menilai pemerintah perlu memperkuat literasi digital "
            "masyarakat untuk menangkal penyebaran berita hoaks di media sosial."
        )
        text = st.text_area(
            "Teks Berita",
            value=default_text,
            height=260,
            help="Masukkan berita dalam bahasa Indonesia. Minimal beberapa kalimat.",
        )

        cek = st.button("Cek Berita")

    with col_result:

        if cek:
            label, proba_hoax = predict_text(text)

            if label == "N/A":
                st.write("Masukkan teks berita terlebih dahulu.")
            else:
                # Label badge
                if label == "Real":
                    st.markdown(
                        """
                        <div class="metric-label">Hasil Prediksi</div>
                        <div style="margin-top:0.65rem;">
                            <span class="label-real">
                                <span class="label-dot"></span>
                                Berita cenderung <b>REAL</b>
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """
                        <div class="metric-label">Hasil Prediksi</div>
                        <div style="margin-top:0.65rem;">
                            <span class="label-hoax">
                                <span class="label-dot"></span>
                                Berita cenderung <b>HOAKS</b>
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("<br/>", unsafe_allow_html=True)

                # Probability display
                pct_hoax = proba_hoax * 100
                pct_real = 100 - pct_hoax

                st.markdown(
                    f"""
                    <div class="metric-label">Probabilitas Hoaks</div>
                    <div class="metric-value">{pct_hoax:.1f}%</div>
                    <div style="font-size:0.82rem; ">
                        (Probabilitas bahwa berita termasuk kelas <b>Hoaks</b> menurut model.)
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.progress(min(max(proba_hoax, 0.0), 1.0))

                st.markdown("---")

                st.markdown(
                    f"""
                    <div style="font-size:0.86rem;">
                        <b>Interpretasi cepat:</b><br/>
                        • Jika skor hoaks &gt; 70%, sebaiknya sangat berhati-hati dan lakukan cross-check. <br/>
                        • Jika skor berada di sekitar 50%, berita mengandung pola bahasa campuran — verifikasi manual sangat disarankan. <br/>
                        • Jika skor hoaks &lt; 30%, berita cenderung mengikuti pola bahasa berita <i>real</i> dalam dataset pelatihan.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
          st.markdown(
              """
              <div style="
                  margin-top:1rem;
                  padding:1.2rem 1.4rem;
                  border:1px #374151;
                  border-radius:14px;
                  background:rgba(255,255,255,0.02);
                  text-align:center;
                  font-size:0.9rem;
              ">
                  ⏳ <b>Belum ada prediksi.</b><br/>
                  Masukkan teks berita di sebelah kiri lalu klik tombol <b>Cek Berita</b>.
              </div>
              """,
              unsafe_allow_html=True,
            )


        st.markdown("</div>", unsafe_allow_html=True)

elif page == "Ciri-ciri Berita Hoaks":
    st.markdown('<span class="pill">Edukasi</span>', unsafe_allow_html=True)
    st.markdown("<h1>Ciri-ciri Umum Berita Hoaks</h1>", unsafe_allow_html=True)

    st.markdown(
        """
        Meskipun model machine learning dapat membantu, kemampuan literasi digital pengguna tetap
        yang paling penting. Berikut beberapa ciri umum berita hoaks di media online:
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="card">
            <h3>🧨 Judul Sensasional & Provokatif</h3>
            <ul>
                <li>Banyak huruf kapital: <i>"HEBOH!!!"</i>, <i>"TERBONGKAR!!"</i></li>
                <li>Memancing emosi marah / takut berlebihan.</li>
                <li>Sering tidak seimbang dengan isi berita.</li>
            </ul>

            <h3>❓ Sumber Tidak Jelas</h3>
            <ul>
                <li>Tidak menyebutkan narasumber yang kredibel.</li>
                <li>Domain situs mencurigakan atau mirip media resmi.</li>
                <li>Tidak ada informasi tanggal, penulis, atau redaksi.</li>
            </ul>

            <h3>📷 Konten Visual Menyesatkan</h3>
            <ul>
                <li>Foto/video lama dipakai untuk konteks kejadian baru.</li>
                <li>Gambar diambil dari peristiwa berbeda.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="card">
            <h3>📤 Ajakan Share Berantai</h3>
            <ul>
                <li>Kalimat seperti <i>"Sebarkan ke semua grup!"</i>, <i>"Jangan berhenti di kamu"</i>.</li>
                <li>Menekankan kecepatan share daripada verifikasi.</li>
            </ul>

            <h3>🧪 Tidak Konsisten dengan Fakta Lain</h3>
            <ul>
                <li>Bertolak belakang dengan informasi dari media kredibel.</li>
                <li>Tidak ada rilis resmi dari lembaga terkait.</li>
            </ul>

            <h3>💬 Teknik Argumentasi Lemah</h3>
            <ul>
                <li>Banyak opini pribadi yang dibungkus seolah fakta.</li>
                <li>Menggunakan kata-kata absolut: <i>"pasti", "dijamin", "100%"</i>.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Cara Verifikasi Sederhana")
    st.markdown(
        """
        - Cari berita yang sama di beberapa media arus utama.  
        - Gunakan situs cek fakta seperti **TurnBackHoax.id**.  
        - Periksa tanggal, konteks, dan sumber resmi (pemerintah, lembaga terkait).  
        - Waspadai screenshot chat / status yang mudah dimanipulasi.
        """
    )

elif page == "Sumber Data & Metode":
    st.markdown('<span class="pill">Metodologi</span>', unsafe_allow_html=True)
    st.markdown("<h1>Sumber Data & Metode Pemodelan</h1>", unsafe_allow_html=True)

    st.markdown("## 📂 Dataset")
    st.markdown(
        """
        Aplikasi ini menggunakan dataset **“Deteksi Berita Hoaks Indo”** dari Kaggle
        yang disusun oleh **Mochamad Abdul Azis**.  
        Dataset berisi berita dalam bahasa Indonesia yang diambil dari:
        """
    )

    st.markdown(
        """
        - **TurnBackHoax.id** – basis data hoaks yang dikurasi oleh MAFINDO  
        - **AntaraNews** – kantor berita nasional  
        - **Kompas**  
        - **Detik**  
        """
    )

    st.markdown(
        """
        Perkiraan jumlah entri:
        - TurnBackHoax.id: ~12.744  
        - Antaranews: ~4.200  
        - Kompas: ~3.500  
        - Detik: ~3.498  

        Label utama:
        - `0` → Real (berita benar / bukan hoaks)  
        - `1` → Hoax  
        """
    )

    st.markdown("## 🧠 Metode Machine Learning")
    st.markdown(
        """
        1. **Pre-processing Teks**  
           - Normalisasi huruf kecil  
           - Pembersihan tanda baca, angka, dan karakter tidak relevan  
           - Tokenisasi dasar  

        2. **Representasi Teks – TF-IDF**  
           - Menggunakan `TfidfVectorizer` dari scikit-learn  
           - N-gram: 1–2 (unigram & bigram)  
           - `min_df=5`, `max_df=0.9`  

        3. **Model Klasifikasi – Logistic Regression**  
           - `LogisticRegression(max_iter=200, n_jobs=-1)`  
           - Fungsi prediksi:
             - `predict()` → kelas hoaks / real  
             - `predict_proba()` → probabilitas masing-masing kelas  

        4. **Evaluasi**  
           - Train / validation / test split  
           - Metrik utama:
             - Accuracy (test): ≈ **0.979**  
             - F1-score seimbang untuk kelas Real & Hoaks  
             - AUC ROC (test): ≈ **0.998**  
             - Confusion matrix menunjukkan kesalahan relatif kecil pada kedua kelas.
        """
    )

    st.markdown("## ⚖️ Lisensi & Catatan")
    st.markdown(
        """
        - Dataset hanya digunakan untuk **riset dan edukasi**, bukan untuk tujuan komersial.  
        - Hak cipta konten berita tetap milik masing-masing media.  
        - Jika digunakan dalam laporan atau publikasi, harap mencantumkan atribusi kepada:
            - Sumber berita asli (Antaranews, Kompas, Detik, TurnBackHoax.id).  
            - Pembuat dataset: **Mochamad Abdul Azis** (Kaggle).  
        """
    )

elif page == "Tentang Tim":
    st.markdown('<span class="pill">About</span>', unsafe_allow_html=True)
    st.markdown("<h1>Tentang Proyek & Tim Pengembang</h1>", unsafe_allow_html=True)

    st.markdown(
        """
        Aplikasi ini dikembangkan sebagai bagian dari tugas/penelitian kelompok di bidang
        **Natural Language Processing dan Keamanan Informasi / Literasi Digital**.
        """
    )

    st.markdown("## 🎯 Tujuan Proyek")
    st.markdown(
        """
        - Meningkatkan kesadaran mengenai bahaya penyebaran berita hoaks.  
        - Menunjukkan bagaimana teknik **machine learning** dapat membantu proses cek fakta awal.  
        - Menjadi studi kasus penerapan **TF-IDF + Logistic Regression** pada teks bahasa Indonesia.  
        """
    )

    st.markdown("## 👥 Anggota Kelompok")
    st.markdown(
        """
        - **Taufik Qurohman**
        - **Muhammad Zikra Al Rizkya Adler**
        - **Ariel Saradilla**
        - **M. Agung Ramadhan**
        - **Hauzan Rafi Attallah**
        """
    )

    st.markdown("## 📌 Catatan Penggunaan")
    st.markdown(
        """
        - Aplikasi ini tidak menggantikan kerja jurnalis dan pemeriksa fakta profesional.  
        - Hasil prediksi sebaiknya selalu dikombinasikan dengan:
          - pengecekan ke media resmi,  
          - rilis lembaga pemerintah / otoritas terkait,  
          - dan platform cek fakta independen.
        """
    )
