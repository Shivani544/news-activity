import pandas as pd
import os
import glob
from textblob import TextBlob

# Load processed dataset
# Try to use the most recent backup if available
csv_files = sorted(glob.glob("data/processed_news*.csv"), key=os.path.getmtime, reverse=True)
csv_file = csv_files[0] if csv_files else "data/processed_news.csv"

df = pd.read_csv(csv_file)

print(f"Dataset loaded from: {csv_file}")
print("Dataset loaded successfully!")
print(df[["processed_text"]].head())

# Sentiment analysis function
def get_sentiment(text):
    analysis = TextBlob(str(text))
    polarity = analysis.sentiment.polarity
    
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis
df["sentiment"] = df["processed_text"].apply(get_sentiment)

# Save final dataset
df.to_csv("data/news_with_sentiment.csv", index=False)

print("\nSentiment analysis completed successfully!")

# ✅ THIS WAS MISSING
print("\nSentiment Distribution:")
print(df["sentiment"].value_counts())