import pandas as pd
import re

# STEP 2: Load cleaned dataset (from Milestone 1)
df = pd.read_csv("data/news_data_cleaned.csv")

print("Dataset loaded successfully!")
print(df.head())

# STEP 2: Combine Title + Description into one text column
df["news_text"] = df["Title"].fillna("") + " " + df["Description"].fillna("")

# STEP 3: Text cleaning function (NLP style)
def clean_text(text):
    text = str(text).lower()                 # convert to lowercase
    text = re.sub(r'<.*?>', '', text)        # remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove punctuation & special characters
    text = re.sub(r'\s+', ' ', text)         # remove extra spaces
    return text.strip()

# Apply cleaning function
df["cleaned_news"] = df["news_text"].apply(clean_text)

# Save processed dataset
df.to_csv("data/processed_news.csv", index=False)

# print("\nSTEP 2 & STEP 3 completed successfully!")
print("New file created: processed_news.csv")