import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

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

# Convert text to TF-IDF
vectorizer = TfidfVectorizer(
    max_features=1000,
    stop_words="english"
)

tfidf_matrix = vectorizer.fit_transform(df["processed_text"])

print("\nTF-IDF Matrix Shape:", tfidf_matrix.shape)

# Train LDA model (3 topics)
lda_model = LatentDirichletAllocation(
    n_components=3,      # number of topics (3–5 allowed)
    random_state=42
)

lda_model.fit(tfidf_matrix)

# Get feature names
feature_names = vectorizer.get_feature_names_out()

# Extract top words per topic
num_top_words = 10

print("\nGenerated Topics:\n")

for topic_idx, topic in enumerate(lda_model.components_):
    print(f"Topic {topic_idx + 1} → ", end="")
    
    top_indices = topic.argsort()[-num_top_words:][::-1]
    
    top_words = [feature_names[i] for i in top_indices]
    print(", ".join(top_words))