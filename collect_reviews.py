"""
Collect Steam reviews for one game via Steam's public review endpoint (no API key).
Samples positive and negative reviews separately so the dataset is balanced.
Usage:  python collect_reviews.py [app_id] [reviews_per_class]
Default: Pacify (967050), 1000 per class -> reviews.csv
Find an app_id in any Steam store URL: store.steampowered.com/app/<app_id>/
"""
import sys
import time

import pandas as pd
import requests

APP_ID = sys.argv[1] if len(sys.argv) > 1 else "967050"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
URL = f"https://store.steampowered.com/appreviews/{APP_ID}"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}


def fetch(params):
    """GET with retries + backoff; returns None if Steam keeps timing out."""
    for attempt in range(6):
        try:
            return requests.get(URL, params=params, headers=HEADERS, timeout=60).json()
        except (requests.exceptions.RequestException, ValueError):
            wait = 3 * (attempt + 1)
            print(f"\n  timeout/error, retry {attempt + 1}/6 in {wait}s ...")
            time.sleep(wait)
    return None


def collect(review_type, label):
    rows, seen, cursor = [], set(), "*"
    while len(rows) < N:
        r = fetch(dict(
            json=1, filter="recent", language="english", review_type=review_type,
            purchase_type="all", num_per_page=100, cursor=cursor))
        if r is None:
            print(f"\n  giving up on {review_type} after repeated failures; keeping {len(rows)} collected")
            break
        batch = r.get("reviews", [])
        if not batch:
            break
        for x in batch:
            if x["recommendationid"] in seen or len(x["review"].strip()) < 15:
                continue
            seen.add(x["recommendationid"])
            rows.append({
                "text": x["review"].replace("\n", " ").strip(),
                "label": label,                    # 1 = Recommended, 0 = Not Recommended
                "playtime_hrs": round(x["author"]["playtime_forever"] / 60, 1),
                "timestamp": x["timestamp_created"],
            })
        cursor = r["cursor"]
        print(f"{review_type}: {len(rows)}", end="\r")
        time.sleep(1.5)
    return rows[:N]


df = pd.DataFrame(collect("positive", 1) + collect("negative", 0))
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("reviews.csv", index=False)
print(f"\nSaved {len(df)} reviews -> reviews.csv  (positive: {df.label.sum()}, negative: {(df.label == 0).sum()})")