import requests

api_key = "71d71c72d194432490a8e95668fe77ec"

url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={api_key}"

response = requests.get(url)
data = response.json()

articles = data["articles"]

print("Top News Headlines:\n")

for i in range(10):   # show 10 titles
    print(f"{i+1}. {articles[i]['title']}")
