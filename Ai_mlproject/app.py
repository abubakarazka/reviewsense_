import streamlit as st
import pandas as pd
import joblib
import os
import plotly.graph_objects as go
from collections import Counter
import numpy as np

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="ReviewSense",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM CSS (PINK MODERN)
# =========================
st.markdown("""
<style>

/* =====================
GLOBAL
===================== */
html, body {
    background-color: #faf6f8;
    color: #333;
    font-family: 'Segoe UI', sans-serif;
}

/* =====================
CARD
===================== */
.card {
    background: #ffffff;
    padding: 22px;
    border-radius: 16px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.06);
    margin-bottom: 20px;
}

/* =====================
TEXT
===================== */
.section-title {
    font-size: 22px;
    font-weight: 600;
    color: #4a2c37;
}

.metric-title {
    font-size: 13px;
    color: #777;
}

.metric-value {
    font-size: 30px;
    font-weight: 700;
    color: #b84d6f;
}

/* =====================
SIDEBAR
===================== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #7a3f52, #5c2f3c);
}

section[data-testid="stSidebar"] * {
    color: #f5f5f5;
}

/* =====================
FILE UPLOADER FIX
===================== */
section[data-testid="stSidebar"] div[data-testid="stFileUploader"] {
    background: #ffffff;
    border-radius: 14px;
    padding: 10px;
}

section[data-testid="stSidebar"] div[data-testid="stFileUploader"] * {
    color: #333 !important;
}

/* upload button */
section[data-testid="stSidebar"] button {
    background-color: #b84d6f !important;
    color: white !important;
    border-radius: 10px;
    font-weight: 500;
}

/* =====================
SUCCESS / WARNING
===================== */
.stAlert-success {
    background-color: #e8f5e9;
    color: #256029;
}

.stAlert-warning {
    background-color: #fff8e1;
    color: #7a5d00;
}

/* =====================
TABLE
===================== */
thead tr th {
    background-color: #f4e6ec !important;
    color: #4a2c37 !important;
}

tbody tr td {
    font-size: 14px;
}

/* =====================
SCROLLBAR
===================== */
::-webkit-scrollbar {
    width: 6px;
}
::-webkit-scrollbar-thumb {
    background: #c58aa0;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)



# =========================
# SIDEBAR
# =========================
st.sidebar.markdown("""
<h2 style='color:#ff4d8d'>💗 ReviewSense</h2>
<p style='color:gray'>AI Review & Reputation Dashboard</p>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
BASE_DIR = os.path.dirname(__file__)
try:
    model = joblib.load(os.path.join(BASE_DIR, "model", "sentiment_model.pkl"))
    vectorizer = joblib.load(os.path.join(BASE_DIR, "model", "vectorizer.pkl"))
    st.sidebar.success("✅ Model berhasil dimuat")
except Exception as e:
    st.sidebar.error(f"❌ Error loading model: {e}")
    st.stop()

# =========================
# UPLOAD CSV
# =========================
st.sidebar.markdown("---")
st.sidebar.markdown("### 📂 Upload Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV (wajib ada kolom: review_text)", 
    type=["csv"],
    help="CSV harus memiliki kolom 'review_text'. Kolom 'sentiment' opsional untuk perbandingan."
)

if not uploaded_file:
    st.markdown("""
    <div class="card">
        <h2>🎯 Cara Menggunakan ReviewSense</h2>
        <ol>
            <li>Upload file CSV dengan kolom <code>review_text</code> (wajib)</li>
            <li>Kolom <code>product_name</code> opsional (untuk filter produk)</li>
            <li>Model AI akan otomatis memprediksi sentimen</li>
            <li>Lihat hasil analisis dan confidence score</li>
        </ol>
        <p><b>Format CSV Minimal:</b></p>
        <pre>
review_text
"Produk sangat bagus dan cepat sampai"
"Kualitas jelek, mengecewakan"
"Biasa saja"
        </pre>
        <p><b>Format CSV dengan Produk (opsional):</b></p>
        <pre>
review_text,product_name
"Produk sangat bagus!",Produk A
"Pengiriman lambat",Produk B
        </pre>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# =========================
# LOAD & VALIDATE DATA
# =========================
try:
    df = pd.read_csv(uploaded_file, engine="python")
except Exception as e:
    st.error(f"❌ Error membaca CSV: {e}")
    st.stop()

# Auto-detect kolom review text
possible_review_cols = ['review_text', 'content', 'review', 'ulasan', 'text', 'komentar', 'comment']
review_col = None

for col in possible_review_cols:
    if col in df.columns:
        review_col = col
        break

if review_col is None:
    st.error("❌ Kolom review tidak ditemukan di CSV")
    st.info(f"💡 CSV harus memiliki salah satu kolom: {', '.join(possible_review_cols)}")
    st.info(f"📋 Kolom yang tersedia di CSV Anda: {', '.join(df.columns.tolist())}")
    st.stop()

# Rename ke review_text untuk konsistensi
if review_col != 'review_text':
    df['review_text'] = df[review_col]
    st.sidebar.success(f"✅ Kolom '{review_col}' terdeteksi sebagai review text")

# Deteksi apakah ada label asli
has_original_labels = "sentiment" in df.columns
if has_original_labels:
    df.rename(columns={"sentiment": "sentiment_original"}, inplace=True)
    st.sidebar.info("ℹ️ Label asli terdeteksi - akan dibandingkan dengan prediksi")

# Deteksi apakah ada product_name
has_product_name = "product_name" in df.columns
if not has_product_name:
    df["product_name"] = "Semua Review"
    st.sidebar.info("ℹ️ Kolom 'product_name' tidak ada - menampilkan semua review")

# =========================
# SENTIMENT PREDICTION
# =========================
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Prediksi Sentimen")

with st.spinner("🔄 Model sedang menganalisis review..."):
    # Transform text
    X = vectorizer.transform(df["review_text"].astype(str).fillna(""))
    
    # Predict sentiment
    df["sentiment_predicted"] = model.predict(X)
    
    # Get confidence scores (probability)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        df["confidence"] = np.max(proba, axis=1) * 100
    else:
        # Jika model tidak support predict_proba
        df["confidence"] = 85.0  # default confidence
    
st.sidebar.success(f"✅ {len(df)} review berhasil dianalisis")

# Gunakan prediksi model sebagai hasil utama
df["sentiment"] = df["sentiment_predicted"]

# =========================
# SIDEBAR FILTER (hanya jika ada product_name)
# =========================
if has_product_name:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Filter Produk")
    products = ["SEMUA PRODUK"] + sorted(df["product_name"].unique().tolist())
    selected_product = st.sidebar.selectbox("Pilih Produk", products)
    filtered_df = df if selected_product == "SEMUA PRODUK" else df[df["product_name"] == selected_product]
else:
    selected_product = "Semua Review"
    filtered_df = df

# =========================
# METRICS
# =========================
count = filtered_df["sentiment"].value_counts()
pos = count.get("positive", 0)
neu = count.get("neutral", 0)
neg = count.get("negative", 0)
total = pos + neu + neg
health = round((pos / total) * 100, 1) if total else 0

avg_confidence = round(filtered_df["confidence"].mean(), 1)

# =========================
# HEADER
# =========================
st.markdown(f"""
<div class="card">
    <div class="section-title">🛍️ Dashboard Reputasi - Hasil Prediksi AI</div>
    <p style="color:gray">Produk: <b>{selected_product}</b> | Total Review: <b>{total}</b> | Avg Confidence: <b>{avg_confidence}%</b></p>
</div>
""", unsafe_allow_html=True)

# =========================
# KPI CARDS
# =========================
c1, c2, c3, c4 = st.columns(4)

def metric(col, title, value, emoji):
    with col:
        st.markdown(f"""
        <div class="card">
            <div class="metric-title">{emoji} {title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

metric(c1, "Positive (Prediksi)", pos, "😊")
metric(c2, "Neutral (Prediksi)", neu, "😐")
metric(c3, "Negative (Prediksi)", neg, "😡")
metric(c4, "Health Score", f"{health}%", "❤️")

# =========================
# COMPARISON IF LABELS EXIST
# =========================
if has_original_labels:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📊 Perbandingan: Label Asli vs Prediksi Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        original_count = filtered_df["sentiment_original"].value_counts()
        st.markdown("**Label Asli (dari CSV):**")
        st.write(original_count)
    
    with col2:
        predicted_count = filtered_df["sentiment_predicted"].value_counts()
        st.markdown("**Prediksi Model AI:**")
        st.write(predicted_count)
    
    # Calculate accuracy
    if "sentiment_original" in filtered_df.columns:
        accuracy = (filtered_df["sentiment_original"] == filtered_df["sentiment_predicted"]).mean() * 100
        st.metric("🎯 Akurasi Model", f"{accuracy:.1f}%")
    
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# PIE CHART + INSIGHT
# =========================
left, right = st.columns([2,1])

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📈 Distribusi Sentimen (Hasil Prediksi)")
    fig = go.Figure(go.Pie(
        labels=["Positive", "Neutral", "Negative"],
        values=[pos, neu, neg],
        hole=0.6,
        marker=dict(colors=['#00c853', '#ffc107', '#f44336'])
    ))
    fig.update_layout(
        title="Prediksi Model AI",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🤖 Insight AI")
    
    st.metric("📊 Total Review Dianalisis", total)
    st.metric("🎯 Rata-rata Confidence", f"{avg_confidence}%")

    neg_reviews = filtered_df[filtered_df["sentiment"] == "negative"]["review_text"]

    if len(neg_reviews) == 0:
        st.success("🎉 Tidak ada review negatif terdeteksi")
    else:
        st.warning(f"⚠️ Ditemukan {len(neg_reviews)} review negatif")
        words = " ".join(neg_reviews.astype(str)).lower().split()
        common_words = [w[0] for w in Counter(words).most_common(5) if len(w[0]) > 3]
        if common_words:
            st.write("**Kata yang sering muncul:**")
            st.write(", ".join(common_words[:5]))

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# CONFIDENCE DISTRIBUTION
# =========================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### 🎯 Distribusi Confidence Score")

fig_conf = go.Figure()
fig_conf.add_trace(go.Histogram(
    x=filtered_df["confidence"],
    nbinsx=20,
    marker_color='#ff4d8d'
))
fig_conf.update_layout(
    xaxis_title="Confidence Score (%)",
    yaxis_title="Jumlah Review",
    height=300
)
st.plotly_chart(fig_conf, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# =========================
# DETAIL REVIEW WITH CONFIDENCE
# =========================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### 📝 Detail Review + Confidence Score")

# Prepare display dataframe
if has_product_name:
    display_df = filtered_df[["product_name", "review_text", "sentiment_predicted", "confidence"]].copy()
    display_df.columns = ["Produk", "Review Text", "Sentimen (AI)", "Confidence (%)"]
else:
    display_df = filtered_df[["review_text", "sentiment_predicted", "confidence"]].copy()
    display_df.columns = ["Review Text", "Sentimen (AI)", "Confidence (%)"]

# Add color coding for confidence
def confidence_color(val):
    if val >= 80:
        return 'background-color: #c8e6c9'
    elif val >= 60:
        return 'background-color: #fff9c4'
    else:
        return 'background-color: #ffcdd2'

# Display with styling
st.dataframe(
    display_df.style.applymap(confidence_color, subset=['Confidence (%)']),
    use_container_width=True,
    height=400
)

# Add legend
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("🟢 **High Confidence**: ≥80%")
with col2:
    st.markdown("🟡 **Medium Confidence**: 60-79%")
with col3:
    st.markdown("🔴 **Low Confidence**: <60%")

st.markdown("</div>", unsafe_allow_html=True)

# =========================
# SAMPLE PREDICTIONS
# =========================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### 🔬 Contoh Prediksi Model")

sample_df = filtered_df.sample(min(5, len(filtered_df)))

for idx, row in sample_df.iterrows():
    sentiment_emoji = {
        "positive": "😊",
        "neutral": "😐", 
        "negative": "😡"
    }
    
    st.markdown(f"""
    **Review:** {row['review_text'][:150]}...  
    **Prediksi:** {sentiment_emoji.get(row['sentiment_predicted'], '❓')} {row['sentiment_predicted'].upper()}  
    **Confidence:** {row['confidence']:.1f}%
    """)
    st.markdown("---")

st.markdown("</div>", unsafe_allow_html=True)
