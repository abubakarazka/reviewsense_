import re
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords

# ================================
# INIT
# ================================
factory = StemmerFactory()
stop_words = set(stopwords.words('indonesian'))

# Kamus slang sederhana (bisa ditambah)
slang_dict = {
    "gk": "tidak",
    "ga": "tidak",
    "nggak": "tidak",
    "bgt": "banget",
    "bgs": "bagus",
    "brg": "barang",
    "tp": "tapi",
    "udh": "sudah",
    "blm": "belum",
    "pdhl": "padahal",
    "aja": "saja",
    "cepet": "cepat"
}

# ================================
# CLEANING FUNCTION
# ================================
def clean_text(text):
    text = text.lower()                       # case folding
    text = re.sub(r"http\S+", "", text)       # hapus URL
    text = re.sub(r"[^a-z\s]", " ", text)     # hapus angka & simbol
    text = re.sub(r"\s+", " ", text).strip()  # hapus spasi berlebih
    return text

# ================================
# NORMALIZATION (SLANG)
# ================================
def normalize_text(text):
    words = text.split()
    normalized_words = [
        slang_dict[word] if word in slang_dict else word
        for word in words
    ]
    return " ".join(normalized_words)

# ================================
# STOPWORD REMOVAL
# ================================
def remove_stopwords(text):
    words = text.split()
    filtered_words = [
        word for word in words
        if word not in stop_words
    ]
    return " ".join(filtered_words)

# ================================
# FULL PIPELINE
# ================================
def preprocess_text(text):
    text = clean_text(text)
    text = normalize_text(text)
    text = remove_stopwords(text)
    return text

