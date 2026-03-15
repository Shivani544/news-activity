import requests
import pandas as pd

API_KEY = "71d71c72d194432490a8e95668fe77ec"

url = f"https://newsapi.org/v2/top-headlines?country=us&pageSize=30&apiKey={API_KEY}"

response = requests.get(url)
data = response.json()

if data.get("status") != "ok":
    print("API Error:", data.get("message"))
else:
    news_list = []

    for article in data.get("articles", []):
        news_list.append({
            "Title": article.get("title"),
            "Description": article.get("description"),
            "Source": article.get("source", {}).get("name"),
            "Date": article.get("publishedAt")
        })

    df = pd.DataFrame(news_list)
    df.to_csv("data/news_data.csv", index=False)

    print("news_data.csv created successfully!")
