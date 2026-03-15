import glob
import os
import re
import time

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob


def save_with_retry(df, final_file, max_retries=5):
    temp_file = f"{os.path.splitext(final_file)[0]}_temp.csv"
    for attempt in range(max_retries):
        try:
            df.to_csv(temp_file, index=False)
            if os.path.exists(final_file):
                os.remove(final_file)
            os.rename(temp_file, final_file)
            return final_file
        except (PermissionError, OSError):
            if attempt < max_retries - 1:
                print(f"File locked, retrying in 2 seconds... (attempt {attempt + 1}/{max_retries})")
                time.sleep(2)
            else:
                backup_file = f"{os.path.splitext(final_file)[0]}_backup_{int(time.time())}.csv"
                df.to_csv(backup_file, index=False)
                print(f"Warning: Could not overwrite {final_file}. Saved to {backup_file}")
                return backup_file


# 1. Load data
print("Reading cleaned data...")
df = pd.read_csv("data/news_data_cleaned.csv")
print("Dataset loaded successfully")

# 2. Text cleaning
print("\nText cleaning started...")
df["news_text"] = df["Title"].fillna("") + " " + df["Description"].fillna("")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["cleaned_news"] = df["news_text"].apply(clean_text)
print("Text cleaning completed")

# 3. Tokenization + stopwords
print("\nTokenization and stopword removal started...")
nltk.download("punkt_tab")
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    tokens = word_tokenize(str(text))
    filtered_tokens = [
        word for word in tokens
        if word.isalpha()
        and word not in stop_words
        and len(word) > 2
    ]
    return " ".join(filtered_tokens)

df["processed_text"] = df["cleaned_news"].apply(preprocess_text)
processed_path = save_with_retry(df, "data/processed_news.csv")
print(f"Tokenization and stopword removal completed (saved to {processed_path})")

# 4. TF-IDF
print("\nApplying TF-IDF feature extraction...")
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 1))
tfidf_matrix = vectorizer.fit_transform(df["processed_text"])
print("TF-IDF matrix generated successfully")

# 5. Print keywords
feature_names = vectorizer.get_feature_names_out()
mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
top_indices = mean_scores.argsort()[-10:][::-1]
top_keywords = [feature_names[idx] for idx in top_indices]

print("\nTop 10 Trending Keywords:")
for word in top_keywords:
    print(f"- {word}")

# 6. Sentiment
print("\nApplying sentiment analysis...")

def get_sentiment(text):
    analysis = TextBlob(str(text))
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return "Positive"
    if polarity < 0:
        return "Negative"
    return "Neutral"

df["sentiment"] = df["processed_text"].apply(get_sentiment)
sentiment_counts = df["sentiment"].value_counts()
save_with_retry(df, "data/news_with_sentiment.csv")
print("Sentiment analysis completed")

print("\nSentiment Distribution:")
print(sentiment_counts)

# 7. Topic modeling
print("\nApplying topic modeling (LDA)...")
lda_model = LatentDirichletAllocation(n_components=3, random_state=42)
lda_model.fit(tfidf_matrix)

num_top_words = 10
topics = []
for topic in lda_model.components_:
    top_indices = topic.argsort()[-num_top_words:][::-1]
    topics.append(", ".join(feature_names[i] for i in top_indices))

print("Topic modeling completed")

print("\nGenerated Topics:")
for i, topic in enumerate(topics, start=1):
    print(f"Topic {i} -> {topic}")

print("\nMILESTONE 2 COMPLETED SUCCESSFULLY")