import streamlit as st
import joblib
import re
import string
import numpy as np
import pandas as pd

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ==========================================================
# KONFIGURASI HALAMAN
# ==========================================================
# Mengubah layout menjadi 'centered' agar form input tidak melebar
st.set_page_config(
    page_title="Analisis Sentimen Traveloka",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==========================================================
# LOAD MODEL & NLP TOOLS (CACHE UNTUK KECEPATAN DEMO)
# ==========================================================
@st.cache_resource
def load_model():
    model = joblib.load("xgboost_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    return model, tfidf

@st.cache_resource
def load_nlp():
    stemmer = StemmerFactory().create_stemmer()
    stopword = StopWordRemoverFactory().create_stop_word_remover()
    return stemmer, stopword

model, tfidf = load_model()
stemmer, stopword = load_nlp()

# ==========================================================
# FUNGSI PREPROCESSING
# ==========================================================
def cleaning(text):
    text = str(text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"www\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def case_folding(text):
    return text.lower()

def tokenizing(text):
    return text.split()

def stopword_removal(text):
    return stopword.remove(text)

def stemming(text):
    return stemmer.stem(text)

def preprocessing(text):
    hasil = {}
    hasil["Teks Asli"] = text
    hasil["Cleaning"] = cleaning(text)
    hasil["Case Folding"] = case_folding(hasil["Cleaning"])
    hasil["Tokenizing"] = tokenizing(hasil["Case Folding"])
    hasil["Stopword"] = stopword_removal(hasil["Case Folding"])
    hasil["Stemming"] = stemming(hasil["Stopword"])
    return hasil

# ==========================================================
# UI: HEADER
# ==========================================================
st.markdown("<h2 style='text-align: center;'>Analisis Sentimen Ulasan Traveloka</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Klasifikasi ulasan layanan pariwisata menggunakan TF-IDF dan XGBoost.</p>", unsafe_allow_html=True)
st.write("---")

# ==========================================================
# UI: INPUT SECTION
# ==========================================================
ulasan = st.text_area(
    "Masukkan Teks Ulasan:",
    height=120,
    placeholder="Contoh: Hotel sangat bersih, pemandangannya bagus, dan pelayanan sangat ramah."
)
prediksi = st.button("Analisis Sentimen", use_container_width=True, type="primary")

st.write("---")

# ==========================================================
# PROSES & OUTPUT
# ==========================================================
if prediksi:
    if ulasan.strip() == "":
        st.warning("Silakan masukkan teks ulasan terlebih dahulu.")
    else:
        with st.spinner("Sedang memproses teks..."):
            # Proses Data
            hasil = preprocessing(ulasan)
            teks_model = hasil["Stemming"]
            
            # Prediksi
            X = tfidf.transform([teks_model])
            hasil_prediksi = int(model.predict(X)[0])
            probabilitas = model.predict_proba(X)[0]
            confidence = float(np.max(probabilitas) * 100)

        # =====================================================
        # LAYOUT HASIL (METRIC & CHART)
        # =====================================================
        st.subheader("Hasil Klasifikasi")
        
        col_metric, col_chart = st.columns(2)
        
        with col_metric:
            label_sentimen = "POSITIF" if hasil_prediksi == 1 else "NEGATIF"
            st.metric(label="Prediksi Sentimen", value=label_sentimen)
            st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
            
        with col_chart:
            # Membuat dataframe untuk chart probabilitas
            df_chart = pd.DataFrame(
                {"Probabilitas (%)": [probabilitas[0] * 100, probabilitas[1] * 100]},
                index=["Negatif", "Positif"]
            )
            st.bar_chart(df_chart)

        st.divider()

        # =====================================================
        # DETAIL PREPROCESSING & DEBUG
        # =====================================================
        st.subheader("Detail Analisis")
        tab1, tab2 = st.tabs(["Tahapan Preprocessing", "Informasi Model"])
        
        with tab1:
            st.markdown("**1. Cleaning & Case Folding**")
            st.code(hasil["Case Folding"])
            
            st.markdown("**2. Tokenizing**")
            st.write(hasil["Tokenizing"])
            
            st.markdown("**3. Stopword Removal**")
            st.code(hasil["Stopword"])
            
            st.markdown("**4. Stemming**")
            st.code(hasil["Stemming"])

        with tab2:
            st.write("**Data Internal Model:**")
            st.json({
                "Hasil Predict": hasil_prediksi,
                "Probabilitas (Negatif, Positif)": [round(p, 4) for p in probabilitas],
                "Dimensi Fitur (Shape)": list(X.shape),
                "Jumlah Kata Dikenali (NNZ)": X.nnz
            })