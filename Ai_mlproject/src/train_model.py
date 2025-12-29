import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

# ================================
# PATH
# ================================
BASE_DIR = os.path.dirname(__file__)

data_path = os.path.join(
    BASE_DIR, "..", "data", "processed", "reviews_clean.csv"
)

model_dir = os.path.join(BASE_DIR, "..", "model")
os.makedirs(model_dir, exist_ok=True)

# ================================
# LOAD DATA
# ================================
df = pd.read_csv(data_path)

df = df.dropna(subset=['clean_review', 'sentiment_label'])
df = df[df['clean_review'].str.strip() != ""]

X = df['clean_review']
y = df['sentiment_label']

# ================================
# SPLIT DATA
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================================
# TF-IDF VECTORIZATION
# ================================
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ================================
# MODEL 1: NAIVE BAYES
# ================================
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

nb_pred = nb_model.predict(X_test_tfidf)

print("\n===== NAIVE BAYES =====")
print("Accuracy:", accuracy_score(y_test, nb_pred))
print(classification_report(y_test, nb_pred))

# ================================
# MODEL 2: SVM
# ================================
from sklearn.svm import LinearSVC

svm_model = LinearSVC(
    class_weight='balanced',
    random_state=42
)

svm_model.fit(X_train_tfidf, y_train)
y_pred_svm = svm_model.predict(X_test_tfidf)
print("\n===== SVM =====")


from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(
    X_train_tfidf, y_train
)

svm_model.fit(X_train_smote, y_train_smote)
y_pred_svm = svm_model.predict(X_test_tfidf)

# ================================
# SIMPAN MODEL TERBAIK (SVM)
# ================================
joblib.dump(svm_model, os.path.join(model_dir, "sentiment_model.pkl"))
joblib.dump(vectorizer, os.path.join(model_dir, "vectorizer.pkl"))

print("\n✅ Model & Vectorizer berhasil disimpan")
