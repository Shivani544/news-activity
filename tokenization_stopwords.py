import pandas as pd
import nltk
import os
import time
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data (run once)
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load processed dataset
df = pd.read_csv("data/processed_news.csv")

print("Dataset loaded successfully!")
print(df[["cleaned_news"]].head())

# Stopword list
stop_words = set(stopwords.words("english"))

# Tokenization + Stopword removal function
def preprocess_text(text):
    tokens = word_tokenize(str(text))   # tokenize text
    
    filtered_tokens = [
        word for word in tokens
        if word.isalpha()               # keep only alphabets
        and word not in stop_words      # remove stopwords
        and len(word) > 2               # remove very short words
    ]
    
    return " ".join(filtered_tokens)

# Apply preprocessing
df["processed_text"] = df["cleaned_news"].apply(preprocess_text)

# Save updated dataset with retry logic using a temp file
import os
import shutil
max_retries = 5
temp_file = "data/processed_news_temp.csv"
final_file = "data/processed_news.csv"

for attempt in range(max_retries):
    try:
        # Save to temp file first
        df.to_csv(temp_file, index=False)
        # Try to replace the original file
        if os.path.exists(final_file):
            os.remove(final_file)
        os.rename(temp_file, final_file)
        break
    except (PermissionError, OSError) as e:
        if attempt < max_retries - 1:
            print(f"File locked, retrying in 2 seconds... (attempt {attempt + 1}/{max_retries})")
            time.sleep(2)
        else:
            # If all retries fail, save to an alternative location
            alt_file = f"data/processed_news_backup_{int(time.time())}.csv"
            df.to_csv(alt_file, index=False)
            print(f"Warning: Could not overwrite original file. Saved to {alt_file}")
            break

# print("\nSTEP 4 completed successfully!")
print("Column 'processed_text' added to processed_news.csv")