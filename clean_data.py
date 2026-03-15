import pandas as pd
import re

df = pd.read_csv("data/news_data.csv")

# Remove empty rows
df.dropna(inplace=True)

# Remove duplicate news based on Title
df.drop_duplicates(subset="Title", inplace=True)

# Clean unwanted symbols
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', str(text))
    return text.strip()

df["Title"] = df["Title"].apply(clean_text)
df["Description"] = df["Description"].apply(clean_text)

df.to_csv("data/news_data_cleaned.csv", index=False)

print("news_data_cleaned.csv created successfully!")
