import requests

API_KEY = "71d71c72d194432490a8e95668fe77ec"

url = f"https://newsapi.org/v2/top-headlines?country=us&pageSize=15&apiKey={API_KEY}"

response = requests.get(url)
data = response.json()

if data.get("status") != "ok":
    print("API Error:", data.get("message"))
else:
    articles = data.get("articles", [])

    print("\nShowing 15 News Articles:\n")

    for i, article in enumerate(articles, start=1):
        print(f"Article {i}")
        print("Title:", article.get("title"))
        print("Source:", article.get("source", {}).get("name"))
        print("Published Date:", article.get("publishedAt"))
        print("-" * 40)
