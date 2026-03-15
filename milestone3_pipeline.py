import pandas as pd
import re
import nltk
from collections import Counter
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("punkt")
nltk.download("stopwords")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

print("===================================")
print("   NEWS PULSE – MILESTONE 3")
print(" Trend Detection & Sentiment Analysis")
print("===================================")


# ---------------------------------
# STEP 1 — LOAD DATASET
# ---------------------------------

print("\nReading cleaned dataset...")

df = pd.read_csv("data/news_data_cleaned.csv")

print("Dataset loaded successfully")
print("Total articles:", len(df))


# ---------------------------------
# STEP 2 — TEXT CLEANING
# ---------------------------------

print("\nText cleaning started...")

df["news_text"] = df["Title"].fillna("") + " " + df["Description"].fillna("")

def clean_text(text):
    
    text = str(text).lower()
    
    text = re.sub(r"<.*?>", "", text)
    
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()

df["cleaned_text"] = df["news_text"].apply(clean_text)

print("Text cleaning completed")


# ---------------------------------
# STEP 3 — REMOVE DUPLICATES
# ---------------------------------

print("\nRemoving duplicate news...")

before = len(df)

df = df.drop_duplicates()

after = len(df)

print("Duplicates removed:", before - after)
print("Remaining articles:", after)


# ---------------------------------
# STEP 4 — TOKENIZATION + STOPWORDS
# ---------------------------------

print("\nTokenization & stopword removal started...")

stop_words = set(stopwords.words("english"))

def preprocess_text(text):

    tokens = word_tokenize(text)

    tokens = [
        word for word in tokens
        if word.isalpha()
        and word not in stop_words
        and len(word) > 2
    ]

    return " ".join(tokens)

df["processed_text"] = df["cleaned_text"].apply(preprocess_text)

print("Tokenization completed")


# ---------------------------------
# STEP 5 — TREND DETECTION (FREQUENCY)
# ---------------------------------

print("\nDetecting trends using word frequency...")

all_words = " ".join(df["processed_text"]).split()

word_freq = Counter(all_words)

top_words = word_freq.most_common(10)

print("\nTop Frequent Words:")

for word, count in top_words:
    
    print(word, ":", count)


# ---------------------------------
# STEP 6 — TREND DETECTION (TF-IDF)
# ---------------------------------

print("\nApplying TF-IDF trend detection...")

vectorizer = TfidfVectorizer(max_features=20)

X = vectorizer.fit_transform(df["processed_text"])

keywords = vectorizer.get_feature_names_out()

print("\nTop TF-IDF Keywords:")

for word in keywords[:10]:
    
    print("-", word)


# ---------------------------------
# STEP 7 — SENTIMENT SCORE
# ---------------------------------

print("\nCalculating sentiment scores...")

def get_sentiment_score(text):

    analysis = TextBlob(text)

    return analysis.sentiment.polarity

df["sentiment_score"] = df["processed_text"].apply(get_sentiment_score)

print("Sentiment score column created")


# ---------------------------------
# STEP 8 — SENTIMENT LABEL
# ---------------------------------

print("\nClassifying sentiment...")

def label_sentiment(score):

    if score > 0:
        
        return "Positive"
    
    elif score < 0:
        
        return "Negative"
    
    else:
        
        return "Neutral"

df["sentiment_label"] = df["sentiment_score"].apply(label_sentiment)

print("Sentiment classification completed")


# ---------------------------------
# STEP 9 — SENTIMENT DISTRIBUTION
# ---------------------------------

print("\nSentiment Distribution:")

sentiment_counts = df["sentiment_label"].value_counts()

print(sentiment_counts)


# ---------------------------------
# STEP 10 — SAVE FINAL DATASET
# ---------------------------------

df.to_csv("data/milestone3_output.csv", index=False)

print("\nFinal dataset saved → data/milestone3_output.csv")


# ---------------------------------
# FINAL MESSAGE
# ---------------------------------

print("\n===================================")
print(" MILESTONE 3 COMPLETED SUCCESSFULLY ")
print("===================================")