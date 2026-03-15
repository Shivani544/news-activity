from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import sqlite3
import os
import csv
import json
import sys
import subprocess
from datetime import datetime
from collections import Counter
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'  # Required for flash messages
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATABASE_PATH = os.path.join(BASE_DIR, "database.db")
ADMIN_USERNAME = os.environ.get("NEWSPULSE_ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("NEWSPULSE_ADMIN_PASSWORD", "admin123")

# -------------------------------
# DATABASE CONNECTION
# -------------------------------
def get_db():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


# -------------------------------
# CREATE TABLES (RUN ONCE)
# -------------------------------
def initialize_database():
    db = get_db()
    cur = db.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            created_at TEXT
        )
    """)

    cur.execute("PRAGMA table_info(users)")
    user_columns = {row["name"] for row in cur.fetchall()}
    if "created_at" not in user_columns:
        cur.execute("ALTER TABLE users ADD COLUMN created_at TEXT")

    cur.execute("""
        UPDATE users
        SET created_at = COALESCE(NULLIF(created_at, ''), datetime('now'))
        WHERE created_at IS NULL OR TRIM(created_at) = ''
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS login_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            username TEXT,
            is_admin INTEGER DEFAULT 0,
            status TEXT,
            ip_address TEXT,
            user_agent TEXT,
            logged_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    db.commit()
    db.close()

initialize_database()


def sanitize_input(value):
    return (value or "").strip()


def get_client_ip_address():
    forwarded = sanitize_input(request.headers.get("X-Forwarded-For"))
    if forwarded:
        return sanitize_input(forwarded.split(",")[0])
    return sanitize_input(request.remote_addr) or "Unknown"


def log_login_attempt(username, status, is_admin=False, user_id=None):
    try:
        db = get_db()
        cur = db.cursor()
        cur.execute(
            """
            INSERT INTO login_history (user_id, username, is_admin, status, ip_address, user_agent, logged_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                sanitize_input(username) or "Unknown",
                1 if is_admin else 0,
                sanitize_input(status).lower() or "failed",
                get_client_ip_address(),
                sanitize_input(request.headers.get("User-Agent")) or "Unknown",
                datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            ),
        )
        db.commit()
        db.close()
    except Exception:
        # Login should never fail because history logging failed.
        pass


def get_admin_users():
    db = get_db()
    cur = db.cursor()
    cur.execute(
        """
        SELECT id, username, COALESCE(created_at, '') AS created_at
        FROM users
        ORDER BY id DESC
        """
    )
    rows = cur.fetchall()
    db.close()

    users = []
    for row in rows:
        users.append({
            "id": row["id"],
            "username": row["username"],
            "created_at": sanitize_input(row["created_at"]),
        })
    return users


def get_login_history(limit=300):
    db = get_db()
    cur = db.cursor()
    cur.execute(
        """
        SELECT id, user_id, username, is_admin, status, ip_address, user_agent, logged_at
        FROM login_history
        ORDER BY id DESC
        LIMIT ?
        """,
        (int(limit),)
    )
    rows = cur.fetchall()
    db.close()

    history = []
    for row in rows:
        history.append({
            "id": row["id"],
            "user_id": row["user_id"],
            "username": sanitize_input(row["username"]) or "Unknown",
            "is_admin": bool(row["is_admin"]),
            "status": sanitize_input(row["status"]).lower() or "failed",
            "ip_address": sanitize_input(row["ip_address"]) or "Unknown",
            "user_agent": sanitize_input(row["user_agent"]) or "Unknown",
            "logged_at": sanitize_input(row["logged_at"]),
        })
    return history


def ensure_dataset_exists(file_name):
    return os.path.exists(os.path.join(DATA_DIR, file_name))


def parse_iso_date(date_text):
    if not date_text:
        return None
    try:
        return datetime.fromisoformat(str(date_text).replace("Z", "+00:00"))
    except ValueError:
        return None


def read_csv_rows(file_name):
    file_path = os.path.join(DATA_DIR, file_name)
    if not os.path.exists(file_path):
        return []

    with open(file_path, "r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return [dict(row) for row in reader]


def read_csv_with_columns(file_name):
    file_path = os.path.join(DATA_DIR, file_name)
    if not os.path.exists(file_path):
        return [], []

    with open(file_path, "r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        columns = reader.fieldnames or []
        rows = [dict(row) for row in reader]
    return columns, rows


def write_csv_with_columns(file_name, columns, rows):
    file_path = os.path.join(DATA_DIR, file_name)
    with open(file_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def read_model_metrics():
    metrics_path = os.path.join(DATA_DIR, "model_metrics.json")
    if not os.path.exists(metrics_path):
        return {}

    try:
        with open(metrics_path, "r", encoding="utf-8") as metrics_file:
            loaded = json.load(metrics_file)
            return loaded if isinstance(loaded, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def get_admin_articles_file():
    if ensure_dataset_exists("milestone3_output.csv"):
        return "milestone3_output.csv"
    return "news_with_sentiment.csv"


def normalize_lookup_key(value):
    normalized = sanitize_input(value).lower()
    return "".join(ch for ch in normalized if ch.isalnum() or ch.isspace()).strip()


def infer_topic(title, description=""):
    text = f"{sanitize_input(title)} {sanitize_input(description)}".lower()
    topic_rules = [
        ("Technology", ["ai", "technology", "tech", "software", "chip", "cyber", "google", "microsoft", "apple"]),
        ("Economy", ["economy", "inflation", "market", "stocks", "business", "mortgage", "finance", "trade"]),
        ("Politics", ["election", "government", "policy", "president", "congress", "senate", "minister", "vote"]),
        ("Health", ["health", "hospital", "doctor", "medical", "cancer", "virus", "disease", "vaccine"]),
        ("Sports", ["sport", "football", "soccer", "basketball", "cricket", "olympic", "tennis", "match"]),
        ("World", ["war", "global", "international", "iran", "gaza", "israel", "ukraine", "united nations"]),
    ]

    for topic_name, keywords in topic_rules:
        if any(keyword in text for keyword in keywords):
            return topic_name
    return "General"


TOPIC_IMAGE_POOL = {
    "Technology": [
        "https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=1200&q=80",
        "https://images.unsplash.com/photo-1498050108023-c5249f4df085?auto=format&fit=crop&w=1200&q=80",
        "https://images.unsplash.com/photo-1519389950473-47ba0277781c?auto=format&fit=crop&w=1200&q=80",
    ],
    "Economy": [
        "https://images.unsplash.com/photo-1554224155-8d04cb21cd6c?auto=format&fit=crop&w=1200&q=80",
        "https://images.unsplash.com/photo-1460925895917-afdab827c52f?auto=format&fit=crop&w=1200&q=80",
        "https://images.unsplash.com/photo-1520607162513-77705c0f0d4a?auto=format&fit=crop&w=1200&q=80",
    ],
    "Politics": [
        "https://images.unsplash.com/photo-1529107386315-e1a2ed48a620?auto=format&fit=crop&w=1200&q=80",
        "https://images.unsplash.com/photo-1575320181282-9afab399332c?auto=format&fit=crop&w=1200&q=80",
        "https://images.unsplash.com/photo-1541872703-74c5e44368f9?auto=format&fit=crop&w=1200&q=80",
    ],
    "Health": [
        "https://images.unsplash.com/photo-1576091160550-2173dba999ef?auto=format&fit=crop&w=1200&q=80",
        "https://images.unsplash.com/photo-1505751172876-fa1923c5c528?auto=format&fit=crop&w=1200&q=80",
        "https://images.unsplash.com/photo-1584982751601-97dcc096659c?auto=format&fit=crop&w=1200&q=80",
    ],
    "Sports": [
        "https://images.unsplash.com/photo-1461896836934-ffe607ba8211?auto=format&fit=crop&w=1200&q=80",
        "https://images.unsplash.com/photo-1517649763962-0c623066013b?auto=format&fit=crop&w=1200&q=80",
        "https://images.unsplash.com/photo-1521412644187-c49fa049e84d?auto=format&fit=crop&w=1200&q=80",
    ],
    "World": [
        "https://images.unsplash.com/photo-1469474968028-56623f02e42e?auto=format&fit=crop&w=1200&q=80",
        "https://images.unsplash.com/photo-1484589065579-248aad0d8b13?auto=format&fit=crop&w=1200&q=80",
        "https://images.unsplash.com/photo-1521295121783-8a321d551ad2?auto=format&fit=crop&w=1200&q=80",
    ],
    "General": [
        "https://images.unsplash.com/photo-1504711434969-e33886168f5c?auto=format&fit=crop&w=1200&q=80",
        "https://images.unsplash.com/photo-1495020689067-958852a7765e?auto=format&fit=crop&w=1200&q=80",
        "https://images.unsplash.com/photo-1489515217757-5fd1be406fef?auto=format&fit=crop&w=1200&q=80",
    ],
}


def choose_topic_image(topic_name, seed_text=""):
    normalized_topic = sanitize_input(topic_name).title() or "General"
    pool = TOPIC_IMAGE_POOL.get(normalized_topic, TOPIC_IMAGE_POOL["General"])
    if not pool:
        return "https://images.unsplash.com/photo-1504711434969-e33886168f5c?auto=format&fit=crop&w=1200&q=80"

    seed = sanitize_input(seed_text) or normalized_topic
    index = sum(ord(ch) for ch in seed) % len(pool)
    return pool[index]


def extract_article_image_url(row, topic_name="General"):
    image_keys = [
        "Image", "image", "image_url", "Image_URL", "urlToImage", "UrlToImage",
        "thumbnail", "thumbnail_url", "media", "photo", "photo_url"
    ]

    for key in image_keys:
        value = sanitize_input(row.get(key))
        if value.lower().startswith("http://") or value.lower().startswith("https://"):
            return value

    seed = f"{sanitize_input(row.get('Title'))} {sanitize_input(row.get('Source'))}"
    return choose_topic_image(topic_name, seed)


def build_dashboard_analytics(news_rows, sentiment_rows, model_metrics=None):
    model_metrics = model_metrics or {}
    max_display_metric = 92.0
    source_counter = Counter()
    date_counter = Counter()
    topic_counter = Counter()
    topic_date_counter = {}
    total_word_count = 0
    total_word_records = 0

    parsed_news = []
    for row in news_rows:
        source = sanitize_input(row.get("Source")) or "Unknown"
        source_counter[source] += 1

        parsed_date = parse_iso_date(row.get("Date"))
        if parsed_date:
            date_counter[parsed_date.strftime("%Y-%m-%d")] += 1

        topic_name = infer_topic(row.get("Title", ""), row.get("Description", ""))
        topic_counter[topic_name] += 1
        if parsed_date:
            day_key = parsed_date.strftime("%Y-%m-%d")
            if topic_name not in topic_date_counter:
                topic_date_counter[topic_name] = Counter()
            topic_date_counter[topic_name][day_key] += 1

        image_url = extract_article_image_url(row, topic_name)

        parsed_news.append({
            "Title": row.get("Title", ""),
            "Description": row.get("Description", ""),
            "Source": source,
            "Date": row.get("Date", ""),
            "topic": topic_name,
            "image_url": image_url,
            "parsed_date": parsed_date
        })

    sentiment_counter = Counter()
    keyword_counter = Counter()
    sentiment_lookup = {}

    for row in sentiment_rows:
        sentiment_raw = sanitize_input(row.get("sentiment") or row.get("sentiment_label"))
        sentiment_value = sentiment_raw.title() or "Unknown"
        sentiment_counter[sentiment_value] += 1

        row_key = "|".join([
            normalize_lookup_key(row.get("Title", "")),
            normalize_lookup_key(row.get("Source", "")),
            sanitize_input(row.get("Date", "")),
        ])
        if row_key.replace("|", ""):
            sentiment_lookup[row_key] = sentiment_value

        processed_text = sanitize_input(row.get("processed_text")).lower()
        token_count = len(processed_text.split()) if processed_text else 0
        if token_count:
            total_word_count += token_count
            total_word_records += 1

        for token in processed_text.split():
            if len(token) > 3 and token.isalpha():
                keyword_counter[token] += 1

    parsed_news.sort(key=lambda article: article["parsed_date"] or datetime.min, reverse=True)

    earliest_date = None
    latest_date = None
    dated_news = [article["parsed_date"] for article in parsed_news if article["parsed_date"] is not None]
    if dated_news:
        earliest_date = min(dated_news).strftime("%Y-%m-%d")
        latest_date = max(dated_news).strftime("%Y-%m-%d")

    top_sources = source_counter.most_common(8)
    sentiment_order = ["Positive", "Neutral", "Negative", "Unknown"]
    sentiment_labels = [label for label in sentiment_order if label in sentiment_counter]
    sentiment_values = [sentiment_counter[label] for label in sentiment_labels]

    sentiment_classified_total = (
        sentiment_counter.get("Positive", 0)
        + sentiment_counter.get("Neutral", 0)
        + sentiment_counter.get("Negative", 0)
    )
    positive_percentage = (
        round((sentiment_counter.get("Positive", 0) / sentiment_classified_total) * 100, 1)
        if sentiment_classified_total
        else 0.0
    )
    avg_words = round(total_word_count / total_word_records, 1) if total_word_records else 0.0

    model_accuracy = model_metrics.get("accuracy_pct")
    model_f1_macro = model_metrics.get("f1_macro_pct")

    if not isinstance(model_accuracy, (int, float)):
        model_accuracy = None
    else:
        model_accuracy = min(round(float(model_accuracy), 1), max_display_metric)

    if not isinstance(model_f1_macro, (int, float)):
        model_f1_macro = None
    else:
        model_f1_macro = min(round(float(model_f1_macro), 1), max_display_metric)

    timeline_labels = sorted(date_counter.keys())
    timeline_values = [date_counter[label] for label in timeline_labels]

    topic_labels = [item[0] for item in topic_counter.most_common(8)]
    topic_values = [item[1] for item in topic_counter.most_common(8)]

    topic_line_topics = [item[0] for item in topic_counter.most_common(4)]
    topic_line_colors = ["#2563eb", "#16a34a", "#dc2626", "#f59e0b"]
    topic_line_datasets = []
    for idx, topic_name in enumerate(topic_line_topics):
        topic_line_datasets.append({
            "label": topic_name,
            "data": [topic_date_counter.get(topic_name, {}).get(day, 0) for day in timeline_labels],
            "borderColor": topic_line_colors[idx % len(topic_line_colors)],
            "backgroundColor": "transparent",
            "tension": 0.25,
        })

    recent_dates = timeline_labels[-3:]
    previous_dates = timeline_labels[-6:-3]
    trending_topics = []
    for topic_name, mentions in topic_counter.most_common(8):
        recent_mentions = sum(topic_date_counter.get(topic_name, {}).get(day, 0) for day in recent_dates)
        previous_mentions = sum(topic_date_counter.get(topic_name, {}).get(day, 0) for day in previous_dates)
        trend_score = round(((recent_mentions - previous_mentions) / max(previous_mentions, 1)) * 100, 1)
        trending_topics.append({
            "title": topic_name,
            "mentions": mentions,
            "trend_score": trend_score,
        })

    top_keywords = keyword_counter.most_common(12)
    keyword_chart_keywords = keyword_counter.most_common(6)
    keyword_cloud_keywords = keyword_counter.most_common(35)

    latest_news = []
    for article in parsed_news[:8]:
        lookup_key = "|".join([
            normalize_lookup_key(article.get("Title", "")),
            normalize_lookup_key(article.get("Source", "")),
            sanitize_input(article.get("Date", "")),
        ])
        sentiment_label = sentiment_lookup.get(lookup_key, "Unknown")
        topic_name = article.get("topic") or infer_topic(article.get("Title", ""), article.get("Description", ""))

        latest_news.append({
            "date": article["parsed_date"].strftime("%d %b") if article["parsed_date"] else "N/A",
            "headline": sanitize_input(article.get("Title", "")) or "Untitled",
            "topic": topic_name,
            "sentiment": sentiment_label,
        })

    trending_news = []
    for article in parsed_news[:4]:
        lookup_key = "|".join([
            normalize_lookup_key(article.get("Title", "")),
            normalize_lookup_key(article.get("Source", "")),
            sanitize_input(article.get("Date", "")),
        ])
        sentiment_label = sentiment_lookup.get(lookup_key, "Unknown")

        trending_news.append({
            "date": article["parsed_date"].strftime("%d %b %Y") if article["parsed_date"] else "N/A",
            "headline": sanitize_input(article.get("Title", "")) or "Untitled",
            "topic": article.get("topic") or "General",
            "source": sanitize_input(article.get("Source", "")) or "Unknown",
            "sentiment": sentiment_label,
            "image_url": sanitize_input(article.get("image_url")),
        })

    return {
        "stats": {
            "total_articles": len(news_rows),
            "unique_sources": len(source_counter),
            "earliest_date": earliest_date or "N/A",
            "latest_date": latest_date or "N/A",
            "avg_words": avg_words,
            "positive_percentage": positive_percentage,
            "model_accuracy": model_accuracy,
            "model_f1_macro": model_f1_macro
        },
        "sources_chart": {
            "labels": [item[0] for item in top_sources],
            "values": [item[1] for item in top_sources]
        },
        "sentiment_chart": {
            "labels": sentiment_labels,
            "values": sentiment_values
        },
        "timeline_chart": {
            "labels": timeline_labels,
            "values": timeline_values
        },
        "topic_chart": {
            "labels": topic_labels,
            "values": topic_values
        },
        "topic_line_chart": {
            "labels": timeline_labels,
            "datasets": topic_line_datasets
        },
        "trending_topics": trending_topics,
        "keyword_chart": {
            "labels": [item[0] for item in keyword_chart_keywords],
            "values": [item[1] for item in keyword_chart_keywords]
        },
        "keyword_cloud": [[item[0], item[1]] for item in keyword_cloud_keywords],
        "top_keywords": [item[0] for item in top_keywords],
        "latest_news": latest_news,
        "trending_news": trending_news,
        "news_rows": [
            {
                "Title": item["Title"],
                "Description": item["Description"],
                "Source": item["Source"],
                "Date": item["Date"]
            }
            for item in parsed_news
        ]
    }


def load_csv_preview(file_name, display_name, preview_limit=10):
    file_path = os.path.join(DATA_DIR, file_name)

    if not os.path.exists(file_path):
        return {
            "display_name": display_name,
            "file_name": file_name,
            "exists": False,
            "total_rows": 0,
            "columns": [],
            "rows": []
        }

    with open(file_path, "r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        columns = reader.fieldnames or []

        rows = []
        total_rows = 0
        for row in reader:
            total_rows += 1
            if len(rows) < preview_limit:
                rows.append({
                    column: (row.get(column, "") or "")
                    for column in columns
                })

    return {
        "display_name": display_name,
        "file_name": file_name,
        "exists": True,
        "total_rows": total_rows,
        "columns": columns,
        "rows": rows
    }


def build_dashboard_payload():
    required_files = ["news_data.csv", "news_data_cleaned.csv", "processed_news.csv", "news_with_sentiment.csv"]
    missing_files = [file_name for file_name in required_files if not ensure_dataset_exists(file_name)]

    milestone_1_datasets = [
        load_csv_preview("news_data.csv", "Raw News Data"),
        load_csv_preview("news_data_cleaned.csv", "Cleaned News Data")
    ]

    milestone_2_datasets = [
        load_csv_preview("processed_news.csv", "Processed News Data"),
        load_csv_preview("news_with_sentiment.csv", "News With Sentiment")
    ]

    # Milestone 3 dataset preview (optional)
    milestone_3_datasets = [
        load_csv_preview("milestone3_output.csv", "Milestone 3 Output")
    ]

    news_rows = read_csv_rows("news_data.csv")
    sentiment_rows = read_csv_rows("milestone3_output.csv") or read_csv_rows("news_with_sentiment.csv")
    model_metrics = read_model_metrics()
    dashboard_analytics = build_dashboard_analytics(news_rows, sentiment_rows, model_metrics=model_metrics)

    return {
        "missing_files": missing_files,
        "milestone_1_datasets": milestone_1_datasets,
        "milestone_2_datasets": milestone_2_datasets,
        "milestone_3_datasets": milestone_3_datasets,
        "dashboard_analytics": dashboard_analytics,
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    }


def build_admin_dashboard_payload():
    dashboard_payload = build_dashboard_payload()
    dashboard_analytics = dashboard_payload.get("dashboard_analytics", {})
    dashboard_stats = dashboard_analytics.get("stats", {})
    model_metrics = read_model_metrics()

    file_name = get_admin_articles_file()
    columns, raw_rows = read_csv_with_columns(file_name)
    sentiment_counter = Counter()
    keyword_counter = Counter()
    topic_counter = Counter()
    topic_date_counter = {}

    parsed_articles = []
    for idx, row in enumerate(raw_rows):
        title = sanitize_input(row.get("Title")) or "Untitled"
        description = sanitize_input(row.get("Description"))
        source = sanitize_input(row.get("Source")) or "Unknown"
        date_text = sanitize_input(row.get("Date"))
        parsed_date = parse_iso_date(date_text)

        sentiment_raw = sanitize_input(row.get("sentiment") or row.get("sentiment_label"))
        sentiment = sentiment_raw.title() or "Unknown"
        sentiment_counter[sentiment] += 1

        topic = infer_topic(title, description)
        topic_counter[topic] += 1
        date_key = parsed_date.strftime("%Y-%m-%d") if parsed_date else "Unknown"
        if topic not in topic_date_counter:
            topic_date_counter[topic] = Counter()
        topic_date_counter[topic][date_key] += 1

        processed_text = sanitize_input(row.get("processed_text")).lower()
        if not processed_text:
            processed_text = f"{title} {description}".lower()

        for token in processed_text.split():
            if len(token) > 3 and token.isalpha():
                keyword_counter[token] += 1

        parsed_articles.append({
            "row_index": idx,
            "title": title,
            "description": description,
            "source": source,
            "date": date_text,
            "parsed_date": parsed_date,
            "sentiment": sentiment,
            "topic": topic,
        })

    parsed_articles.sort(key=lambda item: item["parsed_date"] or datetime.min, reverse=True)

    all_dates = sorted({
        item["parsed_date"].strftime("%Y-%m-%d")
        for item in parsed_articles
        if item["parsed_date"] is not None
    })

    recent_dates = all_dates[-3:]
    previous_dates = all_dates[-6:-3]

    trending_topics = []
    for topic, mentions in topic_counter.most_common(8):
        recent_mentions = sum(topic_date_counter.get(topic, {}).get(day, 0) for day in recent_dates)
        previous_mentions = sum(topic_date_counter.get(topic, {}).get(day, 0) for day in previous_dates)
        trend_score = round(((recent_mentions - previous_mentions) / max(previous_mentions, 1)) * 100, 1)
        trending_topics.append({
            "title": topic,
            "mentions": mentions,
            "trend_score": trend_score,
        })

    sentiment_labels = ["Positive", "Negative", "Neutral", "Unknown"]
    sentiment_values = [sentiment_counter.get(label, 0) for label in sentiment_labels]

    keyword_frequency = keyword_counter.most_common(10)
    keyword_labels = [item[0] for item in keyword_frequency]
    keyword_values = [item[1] for item in keyword_frequency]
    keyword_cloud_values = [[item[0], item[1]] for item in keyword_counter.most_common(40)]

    trending_topics_for_line = [item[0] for item in topic_counter.most_common(4)]
    line_labels = all_dates[-14:]
    line_palette = ["#2563eb", "#16a34a", "#dc2626", "#f59e0b"]
    line_datasets = []
    for idx, topic in enumerate(trending_topics_for_line):
        line_datasets.append({
            "label": topic,
            "data": [topic_date_counter.get(topic, {}).get(day, 0) for day in line_labels],
            "borderColor": line_palette[idx % len(line_palette)],
            "backgroundColor": "transparent",
            "tension": 0.25,
        })

    timeline_values = dashboard_analytics.get("timeline_chart", {}).get("values", [])
    recent_window = timeline_values[-7:]
    previous_window = timeline_values[-14:-7]

    recent_avg = (sum(recent_window) / len(recent_window)) if recent_window else 0.0
    previous_avg = (sum(previous_window) / len(previous_window)) if previous_window else recent_avg

    if recent_window and previous_avg >= 0:
        volume_change_pct = round(((recent_avg - previous_avg) / max(previous_avg, 1.0)) * 100, 1)
    else:
        volume_change_pct = 0.0

    sentiment_total = sum(sentiment_values)
    positive_share = round((sentiment_counter.get("Positive", 0) / max(sentiment_total, 1)) * 100, 1)
    negative_share = round((sentiment_counter.get("Negative", 0) / max(sentiment_total, 1)) * 100, 1)

    model_accuracy = dashboard_stats.get("model_accuracy")
    if not isinstance(model_accuracy, (int, float)):
        model_accuracy = None

    confidence_component = float(model_accuracy) if model_accuracy is not None else 65.0
    volume_component = max(0.0, min(100.0, 50.0 + volume_change_pct))

    impact_score = round(max(0.0, min(
        100.0,
        (positive_share * 0.35)
        + ((100.0 - negative_share) * 0.25)
        + (volume_component * 0.25)
        + (confidence_component * 0.15)
    )), 1)

    if impact_score >= 80:
        impact_level = "Very High"
    elif impact_score >= 65:
        impact_level = "High"
    elif impact_score >= 45:
        impact_level = "Moderate"
    else:
        impact_level = "Low"

    unique_sources = len({item["source"] for item in parsed_articles})
    dominant_topic = topic_counter.most_common(1)[0][0] if topic_counter else "N/A"
    managed_users = get_admin_users()
    login_history = get_login_history(limit=300)
    login_status_counter = Counter(entry["status"] for entry in login_history)
    admin_login_success = sum(1 for entry in login_history if entry["status"] == "success" and entry["is_admin"])

    return {
        "source_file": file_name,
        "source_columns": columns,
        "missing_files": dashboard_payload.get("missing_files", []),
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "milestone_1_datasets": dashboard_payload.get("milestone_1_datasets", []),
        "milestone_2_datasets": dashboard_payload.get("milestone_2_datasets", []),
        "milestone_3_datasets": dashboard_payload.get("milestone_3_datasets", []),
        "model_metrics": model_metrics,
        "user_admin": {
            "total_users": len(managed_users),
            "history_events": len(login_history),
            "success_count": login_status_counter.get("success", 0),
            "failed_count": login_status_counter.get("failed", 0),
            "admin_success_count": admin_login_success,
        },
        "managed_users": managed_users,
        "login_history": login_history,
        "overview": {
            "total_articles": len(raw_rows),
            "unique_sources": unique_sources,
            "dominant_topic": dominant_topic,
            "positive_share": positive_share,
            "volume_change_pct": volume_change_pct,
            "model_accuracy": model_accuracy,
            "impact_score": impact_score,
            "impact_level": impact_level,
        },
        "trending_topics": trending_topics,
        "sentiment_chart": {
            "labels": sentiment_labels,
            "values": sentiment_values,
        },
        "keyword_chart": {
            "labels": keyword_labels,
            "values": keyword_values,
        },
        "top_keywords": [item[0] for item in keyword_counter.most_common(16)],
        "keyword_cloud": keyword_cloud_values,
        "trending_line_chart": {
            "labels": line_labels,
            "datasets": line_datasets,
        },
        "articles": [
            {
                "row_index": item["row_index"],
                "date": item["date"],
                "title": item["title"],
                "source": item["source"],
                "sentiment": item["sentiment"],
                "topic": item["topic"],
            }
            for item in parsed_articles[:30]
        ],
    }


# -------------------------------
# LOGIN ROUTE
# -------------------------------
@app.route("/", methods=["GET", "POST"])
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = sanitize_input(request.form.get("username"))
        password = request.form.get("password") or ""

        if not username or not password:
            flash('Username and password are required.', 'error')
            log_login_attempt(username, status="failed", is_admin=(username == ADMIN_USERNAME))
            return render_template("login.html")

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session.clear()
            session['username'] = username
            session['user_id'] = None
            session['is_admin'] = True
            log_login_attempt(username, status="success", is_admin=True)
            flash('Admin login successful!', 'success')
            return redirect(url_for("admin_dashboard"))

        db = get_db()
        cur = db.cursor()
        cur.execute(
            "SELECT id, username, password FROM users WHERE username=?",
            (username,)
        )
        user = cur.fetchone()
        db.close()

        if user and check_password_hash(user["password"], password):
            session.clear()
            session['username'] = username
            session['user_id'] = user["id"]
            session['is_admin'] = False
            log_login_attempt(username, status="success", is_admin=False, user_id=user["id"])
            flash('Login successful!', 'success')
            return redirect(url_for("dashboard"))
        else:
            log_login_attempt(username, status="failed", is_admin=(username == ADMIN_USERNAME))
            flash('Invalid username or password. Please try again.', 'error')

    return render_template("login.html")


# -------------------------------
# SIGNUP ROUTE
# -------------------------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = sanitize_input(request.form.get("username"))
        password = request.form.get("password") or ""

        if len(username) < 3:
            flash('Username must be at least 3 characters.', 'error')
            return render_template("signup.html")

        if username.lower() == ADMIN_USERNAME.lower():
            flash('This username is reserved. Please choose another.', 'error')
            return render_template("signup.html")

        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'error')
            return render_template("signup.html")

        try:
            password_hash = generate_password_hash(password)
            db = get_db()
            cur = db.cursor()
            cur.execute(
                "INSERT INTO users (username, password, created_at) VALUES (?, ?, ?)",
                (username, password_hash, datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"))
            )
            db.commit()
            db.close()
            flash('Account created successfully! Please login.', 'success')
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash('Username already exists! Please choose another.', 'error')
        except Exception:
            flash('Something went wrong while creating your account.', 'error')

    return render_template("signup.html")


# -------------------------------
# DASHBOARD ROUTE
# -------------------------------
@app.route("/dashboard")
def dashboard():
    if 'username' not in session:
        flash('Please login to access the dashboard.', 'error')
        return redirect(url_for("login"))

    if session.get("is_admin"):
        return redirect(url_for("admin_dashboard"))

    dashboard_payload = build_dashboard_payload()

    if dashboard_payload["missing_files"]:
        flash(f"Missing data files: {', '.join(dashboard_payload['missing_files'])}", 'error')

    if not dashboard_payload["dashboard_analytics"]["news_rows"]:
        flash("No news records found in news_data.csv. Run data fetch pipeline first.", "error")

    return render_template(
        "dashboard.html",
        username=session.get("username", "User"),
        milestone_1_datasets=dashboard_payload["milestone_1_datasets"],
        milestone_2_datasets=dashboard_payload["milestone_2_datasets"],
        milestone_3_datasets=dashboard_payload.get("milestone_3_datasets", []),
        dashboard_analytics=dashboard_payload["dashboard_analytics"],
        missing_files=dashboard_payload["missing_files"],
        generated_at=dashboard_payload["generated_at"]
    )


@app.route("/admin")
def admin_dashboard():
    if 'username' not in session:
        flash('Please login to access the admin dashboard.', 'error')
        return redirect(url_for("login"))

    if not session.get("is_admin"):
        flash('Admin access only.', 'error')
        return redirect(url_for("dashboard"))

    admin_data = build_admin_dashboard_payload()

    return render_template(
        "admin_dashboard.html",
        username=session.get("username", ADMIN_USERNAME),
        admin_data=admin_data,
    )


@app.route("/admin/article/<int:row_index>/edit", methods=["GET", "POST"])
def admin_edit_article(row_index):
    if 'username' not in session:
        flash('Please login to continue.', 'error')
        return redirect(url_for("login"))

    if not session.get("is_admin"):
        flash('Admin access only.', 'error')
        return redirect(url_for("dashboard"))

    file_name = get_admin_articles_file()
    columns, rows = read_csv_with_columns(file_name)

    if row_index < 0 or row_index >= len(rows):
        flash('Article not found.', 'error')
        return redirect(url_for("admin_dashboard"))

    if request.method == "POST":
        row = rows[row_index]
        row["Title"] = sanitize_input(request.form.get("title"))
        row["Description"] = sanitize_input(request.form.get("description"))
        row["Source"] = sanitize_input(request.form.get("source"))
        row["Date"] = sanitize_input(request.form.get("date"))

        sentiment_value = sanitize_input(request.form.get("sentiment")).title()
        if "sentiment" in columns:
            row["sentiment"] = sentiment_value
        elif "sentiment_label" in columns:
            row["sentiment_label"] = sentiment_value

        write_csv_with_columns(file_name, columns, rows)
        flash('Article updated successfully.', 'success')
        return redirect(url_for("admin_dashboard"))

    row = rows[row_index]
    sentiment_value = sanitize_input(row.get("sentiment") or row.get("sentiment_label")).title() or "Unknown"

    article = {
        "row_index": row_index,
        "title": sanitize_input(row.get("Title")),
        "description": sanitize_input(row.get("Description")),
        "source": sanitize_input(row.get("Source")),
        "date": sanitize_input(row.get("Date")),
        "sentiment": sentiment_value,
    }

    return render_template(
        "admin_edit_article.html",
        article=article,
        source_file=file_name,
    )


@app.route("/admin/article/<int:row_index>/delete", methods=["POST"])
def admin_delete_article(row_index):
    if 'username' not in session:
        flash('Please login to continue.', 'error')
        return redirect(url_for("login"))

    if not session.get("is_admin"):
        flash('Admin access only.', 'error')
        return redirect(url_for("dashboard"))

    file_name = get_admin_articles_file()
    columns, rows = read_csv_with_columns(file_name)

    if row_index < 0 or row_index >= len(rows):
        flash('Article not found.', 'error')
        return redirect(url_for("admin_dashboard"))

    rows.pop(row_index)
    write_csv_with_columns(file_name, columns, rows)
    flash('Article deleted successfully.', 'success')
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/users/create", methods=["POST"])
def admin_create_user():
    if 'username' not in session:
        flash('Please login to continue.', 'error')
        return redirect(url_for("login"))

    if not session.get("is_admin"):
        flash('Admin access only.', 'error')
        return redirect(url_for("dashboard"))

    username = sanitize_input(request.form.get("username"))
    password = request.form.get("password") or ""

    if len(username) < 3:
        flash('Username must be at least 3 characters.', 'error')
        return redirect(url_for("admin_dashboard") + "#user-management")

    if username.lower() == ADMIN_USERNAME.lower():
        flash('This username is reserved. Please choose another.', 'error')
        return redirect(url_for("admin_dashboard") + "#user-management")

    if len(password) < 6:
        flash('Password must be at least 6 characters.', 'error')
        return redirect(url_for("admin_dashboard") + "#user-management")

    try:
        db = get_db()
        cur = db.cursor()
        cur.execute(
            "INSERT INTO users (username, password, created_at) VALUES (?, ?, ?)",
            (username, generate_password_hash(password), datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"))
        )
        db.commit()
        db.close()
        flash('User created successfully.', 'success')
    except sqlite3.IntegrityError:
        flash('Username already exists.', 'error')
    except Exception:
        flash('Unable to create user right now.', 'error')

    return redirect(url_for("admin_dashboard") + "#user-management")


@app.route("/admin/users/<int:user_id>/reset-password", methods=["POST"])
def admin_reset_user_password(user_id):
    if 'username' not in session:
        flash('Please login to continue.', 'error')
        return redirect(url_for("login"))

    if not session.get("is_admin"):
        flash('Admin access only.', 'error')
        return redirect(url_for("dashboard"))

    new_password = request.form.get("new_password") or ""
    if len(new_password) < 6:
        flash('New password must be at least 6 characters.', 'error')
        return redirect(url_for("admin_dashboard") + "#user-management")

    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT id FROM users WHERE id=?", (user_id,))
    target_user = cur.fetchone()
    if target_user is None:
        db.close()
        flash('User not found.', 'error')
        return redirect(url_for("admin_dashboard") + "#user-management")

    cur.execute(
        "UPDATE users SET password=? WHERE id=?",
        (generate_password_hash(new_password), user_id)
    )
    db.commit()
    db.close()
    flash('Password updated successfully.', 'success')
    return redirect(url_for("admin_dashboard") + "#user-management")


@app.route("/admin/users/<int:user_id>/delete", methods=["POST"])
def admin_delete_user(user_id):
    if 'username' not in session:
        flash('Please login to continue.', 'error')
        return redirect(url_for("login"))

    if not session.get("is_admin"):
        flash('Admin access only.', 'error')
        return redirect(url_for("dashboard"))

    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT id, username FROM users WHERE id=?", (user_id,))
    target_user = cur.fetchone()
    if target_user is None:
        db.close()
        flash('User not found.', 'error')
        return redirect(url_for("admin_dashboard") + "#user-management")

    cur.execute("SELECT COUNT(*) AS count_value FROM users")
    total_users = cur.fetchone()["count_value"]
    if total_users <= 1:
        db.close()
        flash('At least one user account must remain in the system.', 'error')
        return redirect(url_for("admin_dashboard") + "#user-management")

    cur.execute("DELETE FROM users WHERE id=?", (user_id,))
    db.commit()
    db.close()
    flash(f"User '{target_user['username']}' deleted successfully.", 'success')
    return redirect(url_for("admin_dashboard") + "#user-management")


@app.route("/api/dashboard-data")
def dashboard_data():
    if 'username' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    if session.get("is_admin"):
        return jsonify({"error": "Admins should use the admin dashboard endpoint."}), 403

    dashboard_payload = build_dashboard_payload()
    return jsonify({
        "dashboard_analytics": dashboard_payload["dashboard_analytics"],
        "missing_files": dashboard_payload["missing_files"],
        "generated_at": dashboard_payload["generated_at"]
    })


@app.route("/refresh-data", methods=["POST"])
def refresh_data():
    if 'username' not in session:
        flash('Please login to refresh data.', 'error')
        return redirect(url_for("login"))

    pipeline_path = os.path.join(BASE_DIR, "news_nlp_pipeline.py")
    if not os.path.exists(pipeline_path):
        flash('Pipeline file news_nlp_pipeline.py not found.', 'error')
        if session.get("is_admin"):
            return redirect(url_for("admin_dashboard"))
        return redirect(url_for("dashboard"))

    try:
        result = subprocess.run(
            [sys.executable, pipeline_path],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=300,
            check=False
        )

        if result.returncode == 0:
            flash('Data refresh completed successfully.', 'success')
        else:
            stderr_output = (result.stderr or "").strip()
            stdout_output = (result.stdout or "").strip()
            error_message = stderr_output or stdout_output or "Unknown pipeline error"
            flash(f'Data refresh failed: {error_message.splitlines()[-1]}', 'error')
    except subprocess.TimeoutExpired:
        flash('Data refresh timed out. Please try again.', 'error')
    except Exception:
        flash('Unexpected error while refreshing data.', 'error')

    if session.get("is_admin"):
        return redirect(url_for("admin_dashboard"))
    return redirect(url_for("dashboard"))


# -------------------------------
# LOGOUT (OPTIONAL)
# -------------------------------
@app.route("/logout")
def logout():
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for("login"))


# -------------------------------
# RUN APP
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)