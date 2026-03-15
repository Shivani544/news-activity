import pandas as pd

df = pd.read_csv("data/news_data_cleaned.csv")

total_articles = len(df)
unique_sources = df["Source"].nunique()

print("Total News Articles:", total_articles)
print("Unique News Sources:", unique_sources)
