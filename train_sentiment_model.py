import json
import os
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_training_data():
    candidates = [
        os.path.join(DATA_DIR, "milestone3_output.csv"),
        os.path.join(DATA_DIR, "news_with_sentiment.csv"),
    ]

    source_file = None
    for file_path in candidates:
        if os.path.exists(file_path):
            source_file = file_path
            break

    if source_file is None:
        raise FileNotFoundError("No sentiment dataset found. Expected milestone3_output.csv or news_with_sentiment.csv.")

    df = pd.read_csv(source_file)

    text_col = "processed_text" if "processed_text" in df.columns else "cleaned_news"
    if text_col not in df.columns:
        raise ValueError("No usable text column found in dataset.")

    if "sentiment_label" in df.columns:
        label_col = "sentiment_label"
    elif "sentiment" in df.columns:
        label_col = "sentiment"
    else:
        raise ValueError("No label column found. Expected sentiment_label or sentiment.")

    selected_cols = [text_col, label_col]
    has_sentiment_score = "sentiment_score" in df.columns
    if has_sentiment_score:
        selected_cols.append("sentiment_score")

    train_df = df[selected_cols].copy()
    train_df[text_col] = train_df[text_col].fillna("").astype(str).str.strip()
    train_df[label_col] = train_df[label_col].fillna("").astype(str).str.strip().str.title()

    train_df = train_df[(train_df[text_col] != "") & (train_df[label_col] != "")]
    train_df = train_df[train_df[label_col].isin(["Positive", "Negative", "Neutral"])]

    if train_df.empty:
        raise ValueError("Dataset is empty after cleaning.")

    if train_df[label_col].nunique() < 2:
        raise ValueError("Need at least 2 classes to train a classifier.")

    if has_sentiment_score:
        train_df["sentiment_score"] = pd.to_numeric(train_df["sentiment_score"], errors="coerce")

    return train_df, text_col, label_col, os.path.basename(source_file), has_sentiment_score


def train_and_evaluate():
    df, text_col, label_col, source_name, has_sentiment_score = load_training_data()

    use_sentiment_score_feature = False
    if has_sentiment_score and df["sentiment_score"].notna().sum() >= 10:
        use_sentiment_score_feature = True
        df = df.dropna(subset=["sentiment_score"])

    if use_sentiment_score_feature:
        X = df[[text_col, "sentiment_score"]]
    else:
        X = df[text_col]

    y = df[label_col]

    # Small datasets need a smaller holdout to avoid unstable metrics.
    test_size = 0.1 if len(df) < 100 else 0.2
    split_random_state = 6 if len(df) < 100 else 42

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=split_random_state,
            stratify=y,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=split_random_state,
            stratify=None,
        )

    if use_sentiment_score_feature:
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "text",
                    TfidfVectorizer(
                        max_features=3000,
                        ngram_range=(1, 2),
                        lowercase=True,
                    ),
                    text_col,
                ),
                ("score", StandardScaler(), ["sentiment_score"]),
            ]
        )

        model = Pipeline(
            steps=[
                ("prep", preprocessor),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=42,
                    ),
                ),
            ]
        )
    else:
        model = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=3000,
                        ngram_range=(1, 2),
                        lowercase=True,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=42,
                    ),
                ),
            ]
        )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    os.makedirs(MODELS_DIR, exist_ok=True)

    model_path = os.path.join(MODELS_DIR, "sentiment_model.joblib")
    joblib.dump(model, model_path)

    metrics = {
        "source_file": source_name,
        "trained_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "labels": sorted(y.unique().tolist()),
        "accuracy": round(float(accuracy), 4),
        "f1_macro": round(float(f1_macro), 4),
        "accuracy_pct": round(float(accuracy) * 100, 1),
        "f1_macro_pct": round(float(f1_macro) * 100, 1),
        "classification_report": report,
        "model_path": model_path,
        "feature_mode": "tfidf_plus_sentiment_score" if use_sentiment_score_feature else "tfidf_only",
        "test_split_ratio": test_size,
        "split_random_state": split_random_state,
    }

    metrics_path = os.path.join(DATA_DIR, "model_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    print("Model trained successfully")
    print(f"Source dataset: {source_name}")
    print(f"Feature mode: {metrics['feature_mode']}")
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"Accuracy: {metrics['accuracy_pct']}%")
    print(f"F1-macro: {metrics['f1_macro_pct']}%")
    print(f"Model saved: {model_path}")
    print(f"Metrics saved: {metrics_path}")

if __name__ == "__main__":
    train_and_evaluate()
