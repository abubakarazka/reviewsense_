import os
import pandas as pd
from preprocessing import preprocess_text

BASE_DIR = os.path.dirname(__file__)

csv_path = os.path.join(
    BASE_DIR, "..", "data", "raw", "tokopedia_product_reviews_2025.csv"
)

df = pd.read_csv(csv_path)

df['clean_review'] = df['review_text'].astype(str).apply(preprocess_text)

output_path = os.path.join(
    BASE_DIR, "..", "data", "processed", "reviews_clean.csv"
)

df[['clean_review', 'sentiment_label']].to_csv(output_path, index=False)

print("✅ Preprocessing selesai")
print(f"📁 Output: {output_path}")
