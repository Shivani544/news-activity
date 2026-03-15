import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load processed dataset
# Try to use the most recent backup if available
import os
import glob

csv_files = sorted(glob.glob("data/processed_news*.csv"), key=os.path.getmtime, reverse=True)
csv_file = csv_files[0] if csv_files else "data/processed_news.csv"

df = pd.read_csv(csv_file)

print(f"Dataset loaded from: {csv_file}")
print("Dataset loaded successfully!")
print(df[["processed_text"]].head())

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    max_features=1000,   # limit vocabulary size
    ngram_range=(1, 1)   # unigrams
)

# Apply TF-IDF
tfidf_matrix = vectorizer.fit_transform(df["processed_text"])

print("\nTF-IDF Matrix Shape:", tfidf_matrix.shape)

# Get feature names (words)
feature_names = vectorizer.get_feature_names_out()

# Calculate average TF-IDF score for each word
mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)

# Get top 10 keyword indices
top_indices = mean_scores.argsort()[-10:][::-1]

# Print top 10 trending keywords
print("\nTop 10 Trending Keywords:\n")

for idx in top_indices:
    print(feature_names[idx])