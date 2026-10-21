
import re
import sys
from pathlib import Path

import pandas as pd

try:
    from langdetect import DetectorFactory, detect
    DetectorFactory.seed = 0
except ImportError:
    detect = None

PER_CLASS = max(1, int(sys.argv[1])) if len(sys.argv) > 1 else 800

INPUT_FILE = Path("reviews.csv")
OUTPUT_FILE = Path("reviews_clean.csv")

BLOCK = re.compile(
    r"\b(sex\w*|horny|cum|tits|boob\w*|gaped?|molest\w*|rape\w*|"
    r"pedo\w*|nsfw|naked|nude|erotic|porn\w*|hentai|orgasm|semen|"
    r"underage|kill myself|suicid\w*)\b",
    re.I,
)

SPAM = re.compile(
    r"leave the (cat|ass)|pet it and give it|\{ ?graphics ?\}|"
    r"graphics\s*\}?-{2,}|this game gave me|achondroplasia|♥{20,}",
    re.I,
)


def is_english(text):
    if detect is None:
        # Fallback when langdetect is unavailable.
        return not any(
            c.isalpha() and not c.isascii()
            for c in text
        )

    try:
        return detect(text) == "en"
    except Exception:
        return False


def is_usable(value):
    text = str(value).strip()
    words = text.split()

    if len(words) < 3 or len(text) > 1500:
        return False

    letters = sum(
        c.isalpha() and c.isascii()
        for c in text
    )

    if letters / max(len(text), 1) < 0.6:
        return False

    if re.search(r"(.)\1{7,}", text):
        return False

    if SPAM.search(text) or BLOCK.search(text):
        return False

    return is_english(text)


def main():
    if not INPUT_FILE.exists():
        raise SystemExit(
            "reviews.csv not found. Run this script "
            "from your project folder."
        )

    df = pd.read_csv(INPUT_FILE)

    if not {"text", "label"}.issubset(df.columns):
        raise SystemExit(
            "reviews.csv must contain text and label columns."
        )

    original_count = len(df)

    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")

    df = df[
        df["label"].isin([0, 1]) & df["text"].ne("")
    ].copy()

    df["label"] = df["label"].astype(int)

    df["_key"] = (
        df["text"]
        .str.lower()
        .str.replace(r"\W+", " ", regex=True)
        .str.strip()
    )

    df = df.drop_duplicates("_key").drop(columns="_key")
    df = df[df["text"].apply(is_usable)].copy()

    counts = df["label"].value_counts()

    if len(counts) < 2:
        raise SystemExit(
            "Not enough reviews in both classes after cleaning."
        )

    per_class = min(PER_CLASS, int(counts.min()))

    df = (
        df.groupby("label", group_keys=False)
        .sample(n=per_class, random_state=42)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    columns = ["text", "label"] + [
        col for col in ("playtime_hrs", "timestamp")
        if col in df.columns
    ]

    df[columns].to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    print(
        f"{original_count} raw -> {len(df)} clean "
        f"({per_class} per class) -> {OUTPUT_FILE}"
    )

    print(
        "Class counts:",
        df["label"].value_counts().sort_index().to_dict(),
    )


if __name__ == "__main__":
    main()
