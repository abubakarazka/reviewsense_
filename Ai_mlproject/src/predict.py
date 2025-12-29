import os
import joblib
from preprocessing import preprocess_text

# ================================
# PATH
# ================================
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")

model_path = os.path.join(MODEL_DIR, "sentiment_model.pkl")
vectorizer_path = os.path.join(MODEL_DIR, "vectorizer.pkl")

# ================================
# LOAD MODEL
# ================================
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

def predict_sentiment(text):
    clean_text = preprocess_text(text)
    vector = vectorizer.transform([clean_text])
    return model.predict(vector)[0]

if __name__ == "__main__":
    print("=== ReviewSense Sentiment Tester ===")
    review = input("Masukkan teks review: ")
    result = predict_sentiment(review)
    print("Prediksi Sentimen:", result)
