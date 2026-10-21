"""Pacify Opinion Mining — Opinion Mining & Tag-Based Categorization.
Run: python -m streamlit run app.py
Input: reviews.csv (created by collect_reviews.py), or upload a CSV with a text column.
"""
import re
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from nltk.corpus import stopwords, wordnet
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from wordcloud import WordCloud

st.set_page_config(page_title="Pacify | Opinion Mining", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

# Palette based on the user's references: cranberry, beige, plum, taupe, ivory.
CRANBERRY = "#3F1116"
PLUM = "#64343C"
TAUPE = "#9D7960"
BEIGE = "#FEEFDC"
IVORY = "#FFF1E9"
INK = "#292321"
MUTED = "#766A65"
LINE = "#E6DAD2"
PALE = "#F8F2EC"
GOLD = "#C28D39"
POSITIVE = "#647C65"
NEGATIVE = "#9C4250"
NEUTRAL = "#A58E80"
SENTIMENT_COLORS = {"Positive": POSITIVE, "Negative": NEGATIVE, "Neutral": NEUTRAL}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap');
:root {{ --cranberry:{CRANBERRY}; --plum:{PLUM}; --taupe:{TAUPE}; --ivory:{IVORY}; --ink:{INK}; --muted:{MUTED}; --line:{LINE}; }}
html, body, [class*="css"] {{ font-family:'DM Sans',sans-serif; color:var(--ink); font-size:17px; }}
.stApp {{ background:#FBF8F4; }}
[data-testid="stHeader"] {{ background:rgba(251,248,244,.96); }}
[data-testid="stSidebar"] {{ background:#F2E8DE; border-right:1px solid #E3D5CA; }}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 {{ color:var(--cranberry); }}
.block-container {{ max-width:1500px; padding-top:1.6rem; padding-bottom:3rem; }}
h1,h2,h3,h4 {{ color:var(--cranberry) !important; letter-spacing:-.025em; }}
h1 {{ font-family:'Playfair Display',serif; font-size:2.65rem !important; font-weight:600; }}
h2 {{ font-family:'Playfair Display',serif; font-size:2rem !important; font-weight:600; }}
h3 {{ font-size:1.35rem !important; font-weight:700; letter-spacing:-.01em; }}
h4 {{ font-size:1.12rem !important; font-weight:700; }}
p, label, .stCaption, [data-testid="stMarkdownContainer"] li {{ color:var(--muted); font-size:1.05rem !important; line-height:1.65; }}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stWidgetLabel"] p {{ font-size:1.02rem !important; line-height:1.5; }}
[data-testid="stMetric"] {{ background:#FFFCF8; border:1px solid var(--line); padding:20px 22px; border-radius:7px; box-shadow:0 2px 7px rgba(63,17,22,.025); }}
[data-testid="stMetricLabel"] {{ color:var(--muted); font-size:1rem !important; }}
[data-testid="stMetricValue"] {{ color:var(--cranberry); font-weight:700; font-size:2.15rem !important; }}
[data-testid="stMetricDelta"] {{ font-size:.95rem !important; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-color:var(--line) !important; }}
.stTabs [data-baseweb="tab-list"] {{ gap:9px; border-bottom:1px solid var(--line); }}
.stTabs [data-baseweb="tab"] {{ background:transparent; color:var(--muted); border-radius:4px 4px 0 0; padding:12px 17px; font-size:1.02rem !important; }}
.stTabs [aria-selected="true"] {{ color:var(--cranberry) !important; border-bottom:3px solid var(--cranberry) !important; }}
.stButton>button, .stDownloadButton>button {{ background:var(--cranberry); color:white; border:1px solid var(--cranberry); border-radius:5px; padding:.65rem 1.05rem; font-size:1rem !important; }}
.stButton>button:hover, .stDownloadButton>button:hover {{ background:var(--plum); border-color:var(--plum); color:white; }}
[data-testid="stDataFrame"] {{ border:1px solid var(--line); border-radius:5px; overflow:hidden; font-size:1rem !important; }}
[data-testid="stDataFrame"] * {{ font-size:14px !important; }}
hr {{ border-color:var(--line); }}
.eyebrow {{ color:var(--plum); font-size:.82rem !important; font-weight:700; letter-spacing:.15em; text-transform:uppercase; margin-bottom:.6rem; }}
.hero {{ background:var(--ivory); border:1px solid #E9D9CF; border-left:5px solid var(--cranberry); padding:27px 32px; margin:0 0 26px 0; }}
.hero-title {{ font-family:'Playfair Display',serif; color:var(--cranberry); font-size:2.45rem; line-height:1.2; margin:0 0 10px 0; }}
.hero-sub {{ color:#6E5D56; font-size:1.08rem !important; margin:0; }}
.section-kicker {{ font-size:.82rem !important; color:var(--taupe); text-transform:uppercase; font-weight:700; letter-spacing:.13em; margin-bottom:6px; }}
.section-intro {{ color:var(--muted); margin-top:-6px; margin-bottom:22px; font-size:1.05rem !important; }}
.note {{ background:#F5ECE3; border-left:3px solid var(--taupe); padding:14px 17px; color:#5F514A; font-size:1rem !important; line-height:1.6; }}
.small-label {{ font-size:.88rem !important; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); font-weight:700; }}
/* Make the top section navigation larger and darker for clear visibility. */
div[role="radiogroup"] label, div[role="tablist"] button {{ font-size:1rem !important; }}
[data-testid="stRadio"] label p {{ font-size:1rem !important; color:var(--ink) !important; }}

/* Clearly show expandable help panels before hover, with a visible chevron and label. */
[data-testid="stExpander"] {{
    background:#F3E8E1 !important;
    border:1px solid #D9C5B9 !important;
    border-radius:7px !important;
    margin:10px 0 18px 0 !important;
}}
[data-testid="stExpander"] summary {{
    color:var(--cranberry) !important;
    font-size:1.02rem !important;
    font-weight:600 !important;
    padding:13px 16px !important;
}}
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span {{
    color:var(--cranberry) !important;
    font-size:1.02rem !important;
}}
[data-testid="stExpander"] summary svg {{
    color:var(--cranberry) !important;
    fill:var(--cranberry) !important;
}}
[data-testid="stExpander"] summary:hover {{
    background:#EAD9CE !important;
}}
/* Hide the redundant footer note and reclaim its bottom whitespace. */
footer {{ visibility:hidden !important; }}
[data-testid="stBottomBlockContainer"] {{ padding-bottom:0 !important; }}
.block-container {{ padding-bottom:1.1rem !important; }}
</style>
""", unsafe_allow_html=True)

# Download NLTK resources only when missing; this avoids noisy repeated output.
@st.cache_resource(show_spinner=False)
def setup_nltk():
    resources = {
        "tokenizers/punkt": "punkt", "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords", "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4", "sentiment/vader_lexicon.zip": "vader_lexicon",
        "taggers/averaged_perceptron_tagger_eng": "averaged_perceptron_tagger_eng",
        "taggers/averaged_perceptron_tagger": "averaged_perceptron_tagger",
    }
    for path, package in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)
    return True

with st.spinner("Preparing text analysis resources…"):
    setup_nltk()

STOP = set(stopwords.words("english")) | {"one", "would", "also", "get", "got", "like", "really", "game", "play", "even", "much"}
LEM, STEM = WordNetLemmatizer(), PorterStemmer()
VADER = SentimentIntensityAnalyzer()

@st.cache_data(show_spinner=False)
def load_default_data():
    for path in ("reviews_clean.csv", "reviews.csv"):
        try:
            data = pd.read_csv(path)
            if "text" in data.columns and not data.empty:
                return data, path
        except FileNotFoundError:
            pass
    return pd.DataFrame(), ""

with st.sidebar:
    st.markdown(f"<div style='padding:7px 0 18px'><div style='font-family:Playfair Display,serif;font-size:1.8rem;color:{CRANBERRY};'>PACIFY</div><div style='font-size:.72rem;letter-spacing:.13em;text-transform:uppercase;color:{MUTED};'>Opinion Mining</div></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**DATASET**")
    uploaded = st.file_uploader("Upload review CSV", type=["csv"], help="Your CSV must contain a column with review text.")
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            source_name = uploaded.name
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
            st.stop()
    else:
        df, source_name = load_default_data()
    if df.empty:
        st.error("No review dataset found. Run `python collect_reviews.py` in the project folder, then refresh.")
        st.stop()
    text_default = list(df.columns).index("text") if "text" in df.columns else 0
    text_col = st.selectbox("Review text column", list(df.columns), index=text_default)
    st.markdown("---")
    st.markdown("**ANALYSIS SETTINGS**")
    max_comp = max(2, min(100, int(df.shape[0] - 1)))
    n_comp = st.slider("LSA components", min_value=2, max_value=max_comp, value=min(50, max_comp))
    neutral_band = st.slider("Neutral sentiment band", 0.0, 0.5, 0.05, 0.01, help="Scores between -band and +band are Neutral.")
    top_k = st.slider("Number of tags", 10, 60, 30, 5)
    st.markdown("---")
    st.caption(f"Source file: `{source_name}`")
    st.caption("Analysis is calculated from the selected dataset in this session.")

raw = df[text_col].fillna("").astype(str).tolist()
valid_indices = [i for i, value in enumerate(raw) if value.strip()]
if len(valid_indices) < 2:
    st.error("Please provide at least two non-empty reviews to analyse.")
    st.stop()
raw = [raw[i] for i in valid_indices]
analysis_df = df.iloc[valid_indices].reset_index(drop=True)

# Stage 1 — cleaning and BoW
def clean(text):
    text = re.sub(r"\[/?[a-z0-9 =]+\]|https?://\S+|www\.\S+|[^a-zA-Z\s]", " ", str(text).lower())
    words = [word for word in word_tokenize(text) if word not in STOP and len(word) > 2]
    return [LEM.lemmatize(word) for word in words]

tokens = [clean(text) for text in raw]
clean_txt = [" ".join(items) if items else "emptyreview" for items in tokens]
stems = [" ".join(STEM.stem(word) for word in items) for items in tokens]

try:
    cv = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    bow = cv.fit_transform(clean_txt)
except ValueError:
    st.error("The selected reviews do not contain enough usable terms after cleaning. Try another text column or dataset.")
    st.stop()
bow_terms = cv.get_feature_names_out()
freq = pd.Series(np.asarray(bow.sum(axis=0)).ravel(), index=bow_terms).sort_values(ascending=False)

# Stage 2 — TF-IDF
min_df = 2 if len(clean_txt) >= 10 else 1
try:
    tv = TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, token_pattern=r"(?u)\b\w+\b")
    X = tv.fit_transform(clean_txt)
except ValueError:
    tv = TfidfVectorizer(ngram_range=(1, 1), min_df=1, token_pattern=r"(?u)\b\w+\b")
    X = tv.fit_transform(clean_txt)
terms = np.array(tv.get_feature_names_out())
mean_tfidf = pd.Series(np.asarray(X.mean(axis=0)).ravel(), index=terms).sort_values(ascending=False)

# Stage 3 — SVD/LSA
n_svd = max(1, min(n_comp, X.shape[1] - 1, X.shape[0] - 1)) if X.shape[1] > 1 and X.shape[0] > 1 else 1
if X.shape[1] > 1 and X.shape[0] > 1:
    svd = TruncatedSVD(n_components=n_svd, random_state=42)
    Z = svd.fit_transform(X)
    explained = float(svd.explained_variance_ratio_.sum())
    cumulative = np.cumsum(svd.explained_variance_ratio_)
else:
    Z = X.toarray()
    explained, cumulative = 0.0, np.array([0.0])

# Stage 4 — three-class opinion classification
compound = np.array([VADER.polarity_scores(text)["compound"] for text in raw])
def to_sentiment(score):
    if score >= neutral_band:
        return "Positive"
    if score <= -neutral_band:
        return "Negative"
    return "Neutral"
res = pd.DataFrame({"Review": raw, "Compound score": compound, "Sentiment": [to_sentiment(s) for s in compound]})
res["Cleaned text"] = clean_txt
sentiment_counts = res["Sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"], fill_value=0)

# Stage 5 — ranked, refined and enriched tags
present_df = np.asarray((X > 0).sum(axis=0)).ravel()
scores = np.asarray(X.sum(axis=0)).ravel() * np.log1p(present_df)
ranked = pd.DataFrame({"Tag candidate": terms, "Ranking score": scores, "Documents present": present_df})
ranked = ranked.sort_values("Ranking score", ascending=False).head(max(top_k * 3, top_k)).reset_index(drop=True)
def pos_ok(tag):
    try:
        return all(pos.startswith(("NN", "JJ")) for _, pos in nltk.pos_tag(tag.split()))
    except LookupError:
        return True
refined = ranked[ranked["Tag candidate"].apply(pos_ok)].copy()
if refined.empty:
    refined = ranked.copy()
refined["_stem"] = refined["Tag candidate"].apply(lambda value: " ".join(STEM.stem(word) for word in value.split()))
refined = refined.groupby("_stem", as_index=False).agg({"Tag candidate": "first", "Ranking score": "sum", "Documents present": "max"})
refined = refined.sort_values("Ranking score", ascending=False).head(top_k).reset_index(drop=True)

def enrich(tag):
    word = tag.split()[-1]
    related = set()
    try:
        for synset in wordnet.synsets(word)[:2]:
            related.update(lemma.name().replace("_", " ") for lemma in synset.lemmas())
            for hypernym in synset.hypernyms()[:1]:
                related.update(lemma.name().replace("_", " ") for lemma in hypernym.lemmas())
    except LookupError:
        pass
    related.discard(word)
    related.discard(tag)
    return sorted(related)[:4]

def dominant_sentiment(tag):
    matched = [res.iloc[i]["Sentiment"] for i, text in enumerate(clean_txt) if tag in text]
    return Counter(matched).most_common(1)[0][0] if matched else "Neutral"

refined["Related tags"] = refined["Tag candidate"].apply(enrich)
refined["Dominant sentiment"] = refined["Tag candidate"].apply(dominant_sentiment)

# Chart styling
def style_plot(fig, height=360, default_title=""):
    # Explicit text colours keep all chart labels readable on the ivory background.
    existing_title = fig.layout.title.text if fig.layout.title and fig.layout.title.text else ""
    title_text = existing_title if existing_title and existing_title.lower() != "undefined" else default_title
    fig.update_layout(
        height=height,
        paper_bgcolor="#FFFCF8",
        plot_bgcolor="#FFFCF8",
        font=dict(family="DM Sans, sans-serif", color=INK, size=14),
        title=dict(
            text=title_text,
            font=dict(family="Playfair Display, serif", size=19, color=CRANBERRY),
            x=0.02, xanchor="left",
        ),
        margin=dict(l=42, r=22, t=58, b=46),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            font=dict(family="DM Sans, sans-serif", color=INK, size=13),
        ),
        xaxis=dict(
            showgrid=True, gridcolor="#E8DDD4", zerolinecolor=LINE, linecolor=LINE,
            tickfont=dict(family="DM Sans, sans-serif", color=INK, size=13),
            title_font=dict(family="DM Sans, sans-serif", color=PLUM, size=14),
        ),
        yaxis=dict(
            showgrid=False, gridcolor="#E8DDD4", zerolinecolor=LINE, linecolor=LINE,
            tickfont=dict(family="DM Sans, sans-serif", color=INK, size=13),
            title_font=dict(family="DM Sans, sans-serif", color=PLUM, size=14),
        ),
    )
    return fig

# Header / overview
st.markdown(f"""
<div class='hero'>
  <div class='eyebrow'>OPINION MINING · USER FEEDBACK ANALYSIS</div>
  <div class='hero-title'>Pacify Review Observatory</div>
  <p class='hero-sub'>Explore what players talk about, how they feel, and which themes dominate the review corpus.</p>
</div>
""", unsafe_allow_html=True)

review_count = len(raw)
unique_terms = len(bow_terms)
unique_tags = len(refined)
avg_length = np.mean([len(text.split()) for text in raw]) if raw else 0
m1, m2, m3, m4 = st.columns(4)
m1.metric("Reviews analysed", f"{review_count:,}")
m2.metric("Unique terms", f"{unique_terms:,}")
m3.metric("Candidate tags", f"{unique_tags:,}")
m4.metric("Avg. review length", f"{avg_length:.1f} words")

# Main workspace navigation
pages = ["Overview", "Text Cleaning & BoW", "TF-IDF Analysis", "Dimensionality Reduction", "Opinion Classification", "Tag Analysis", "Review Explorer"]
selected_page = st.radio("Workspace", pages, horizontal=True, label_visibility="collapsed")
st.markdown("<hr style='margin:0 0 22px 0'>", unsafe_allow_html=True)

if selected_page == "Overview":
    st.markdown("<div class='section-kicker'>CORPUS SUMMARY</div>", unsafe_allow_html=True)
    st.header("Overview")
    st.markdown("<p class='section-intro'>A quick read of the dataset before exploring each stage of the analysis.</p>", unsafe_allow_html=True)
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, hole=.66,
                     color=sentiment_counts.index, color_discrete_map=SENTIMENT_COLORS)
        fig.update_traces(textposition="outside", textinfo="percent", marker=dict(line=dict(color="#FFFCF8", width=3)))
        fig.add_annotation(text=f"{review_count:,}<br><span style='font-size:11px'>reviews</span>", x=.5, y=.5, showarrow=False,
                           font=dict(size=20, color=CRANBERRY, family="Playfair Display, serif"))
        st.plotly_chart(style_plot(fig, 345), use_container_width=True)
        st.caption("Opinion labels are predicted from the review text using VADER compound scores.")
    with right:
        top_freq = freq.head(12).sort_values()
        fig = px.bar(x=top_freq.values, y=top_freq.index, orientation="h", labels={"x":"Occurrences", "y":"Term"},
                     color=top_freq.values, color_continuous_scale=[IVORY, "#C79AA0", CRANBERRY])
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_plot(fig, 345), use_container_width=True)
    st.subheader("Analysis pipeline")
    pipeline = st.columns(5)
    stage_info = [("01", "Clean & count", "Prepare text and build the BoW matrix."), ("02", "TF-IDF", "Represent terms with weighted features."),
                  ("03", "Reduce", "Explore features through SVD/LSA."), ("04", "Classify", "Assign positive, neutral or negative sentiment."),
                  ("05", "Generate tags", "Rank, refine and enrich descriptive terms.")]
    for column, (num, title, desc) in zip(pipeline, stage_info):
        with column:
            st.markdown(f"<div style='border-top:3px solid {PLUM};padding:12px 8px 4px 0'><div style='color:{TAUPE};font-size:.75rem;font-weight:700;letter-spacing:.1em'>{num}</div><div style='font-weight:700;color:{CRANBERRY};margin:6px 0'>{title}</div><div style='font-size:.82rem;color:{MUTED};line-height:1.45'>{desc}</div></div>", unsafe_allow_html=True)

elif selected_page == "Text Cleaning & BoW":
    st.markdown("<div class='section-kicker'>STAGE 01</div>", unsafe_allow_html=True)
    st.header("Text Cleaning & Bag-of-Words")
    st.markdown("<p class='section-intro'>Normalize review text, remove common words, reduce words to their base form, then count term occurrences.</p>", unsafe_allow_html=True)
    a, b = st.columns([1.2, 1], gap="large")
    with a:
        st.subheader("Preprocessing examples")
        examples = pd.DataFrame({"Original review": raw[:8], "Cleaned tokens": [" · ".join(x) for x in tokens[:8]], "Stemmed terms": stems[:8]})
        st.dataframe(examples, use_container_width=True, hide_index=True, height=350)
    with b:
        st.metric("Documents", f"{bow.shape[0]:,}")
        st.metric("Vocabulary dimensions", f"{bow.shape[1]:,}")
        st.metric("Non-zero term counts", f"{bow.nnz:,}")
        st.markdown("<div class='note'>The Bag-of-Words matrix records how often each vocabulary term occurs in each review. Word order is not represented.</div>", unsafe_allow_html=True)
    st.subheader("Most frequent terms")
    chart_data = freq.head(25).sort_values()
    fig = px.bar(x=chart_data.values, y=chart_data.index, orientation="h", title="Most Frequent Terms (Bag-of-Words)", labels={"x":"Total occurrences", "y":"Term"}, color_discrete_sequence=[PLUM])
    st.plotly_chart(style_plot(fig, 560), use_container_width=True)
    with st.expander("Inspect Bag-of-Words matrix"):
        bow_df = pd.DataFrame(bow[:min(30, bow.shape[0])].toarray(), columns=bow_terms)
        st.dataframe(bow_df, use_container_width=True, hide_index=True)

elif selected_page == "TF-IDF Analysis":
    st.markdown("<div class='section-kicker'>STAGE 02</div>", unsafe_allow_html=True)
    st.header("TF-IDF Feature Extraction")
    st.markdown("<p class='section-intro'>Terms are weighted by their importance within a review relative to the corpus. Unigrams and bigrams are included.</p>", unsafe_allow_html=True)
    st.markdown(f"**Feature matrix:** `{X.shape[0]:,} reviews × {X.shape[1]:,} terms` &nbsp; · &nbsp; **Minimum document frequency:** `{min_df}`", unsafe_allow_html=True)
    left, right = st.columns([1.1, .9], gap="large")
    with left:
        top_tfidf = mean_tfidf.head(25).sort_values()
        fig = px.bar(x=top_tfidf.values, y=top_tfidf.index, orientation="h", title="Top Terms by Mean TF-IDF", labels={"x":"Mean TF-IDF", "y":"Term"}, color_discrete_sequence=[PLUM])
        st.plotly_chart(style_plot(fig, 540), use_container_width=True)
    with right:
        doc_i = st.number_input("Review index", min_value=0, max_value=len(raw)-1, value=0, step=1)
        st.markdown(f"<div class='note'>{raw[int(doc_i)]}</div>", unsafe_allow_html=True)
        row = X[int(doc_i)].toarray().ravel()
        top_ids = row.argsort()[::-1][:10]
        inspect = pd.DataFrame({"Term": terms[top_ids], "TF-IDF score": row[top_ids]}).query("`TF-IDF score` > 0")
        st.dataframe(inspect, use_container_width=True, hide_index=True)
    with st.expander("How to interpret TF-IDF"):
        st.write("A higher score indicates that a term is relatively important in this review and not equally common across all reviews. Scores are feature weights, not sentiment scores.")

elif selected_page == "Dimensionality Reduction":
    st.markdown("<div class='section-kicker'>STAGE 03</div>", unsafe_allow_html=True)
    st.header("Dimensionality Reduction")
    st.markdown("<p class='section-intro'>SVD/LSA maps the high-dimensional TF-IDF representation into a smaller feature space for pattern exploration.</p>", unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.metric("Original features", f"{X.shape[1]:,}")
    b.metric("LSA components", f"{Z.shape[1]:,}")
    c.metric("Variance explained", f"{explained:.1%}")
    left, right = st.columns(2, gap="large")
    with left:
        fig = px.line(x=np.arange(1, len(cumulative)+1), y=cumulative, labels={"x":"Number of SVD components", "y":"Cumulative explained variance"}, markers=True)
        fig.update_traces(line=dict(color=PLUM, width=2.5), marker=dict(color=GOLD, size=5))
        st.plotly_chart(style_plot(fig, 390), use_container_width=True)
    with right:
        if Z.shape[1] >= 2 and len(raw) >= 2:
            coords = PCA(n_components=2, random_state=42).fit_transform(Z)
            proj = pd.DataFrame({"Component 1":coords[:,0], "Component 2":coords[:,1], "Review":raw, "Sentiment":[to_sentiment(s) for s in compound]})
            fig = px.scatter(proj, x="Component 1", y="Component 2", color="Sentiment", color_discrete_map=SENTIMENT_COLORS, hover_data=["Review"], opacity=.72)
            st.plotly_chart(style_plot(fig, 390), use_container_width=True)
        else:
            st.info("At least two reduced dimensions are required for the 2D projection.")
    st.markdown("<div class='note'>The 2D projection is for visual exploration. Nearby points have similar reduced representations; separation does not by itself prove a sentiment category.</div>", unsafe_allow_html=True)

elif selected_page == "Opinion Classification":
    st.markdown("<div class='section-kicker'>STAGE 04</div>", unsafe_allow_html=True)
    st.header("Opinion Classification")
    st.markdown("<p class='section-intro'>Each review is assigned a Positive, Neutral or Negative label from its VADER compound score.</p>", unsafe_allow_html=True)
    metrics = st.columns(3)
    for colm, sentiment in zip(metrics, ["Positive", "Neutral", "Negative"]):
        count = int(sentiment_counts[sentiment])
        colm.metric(sentiment, f"{count:,}", f"{count/review_count:.1%} of reviews")
    left, right = st.columns(2, gap="large")
    with left:
        fig = px.bar(x=sentiment_counts.index, y=sentiment_counts.values, color=sentiment_counts.index, color_discrete_map=SENTIMENT_COLORS, labels={"x":"Predicted sentiment", "y":"Reviews"})
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_plot(fig, 350), use_container_width=True)
    with right:
        hist = pd.DataFrame({"Compound score":compound, "Sentiment":res["Sentiment"]})
        fig = px.histogram(hist, x="Compound score", color="Sentiment", color_discrete_map=SENTIMENT_COLORS, nbins=35, barmode="overlay", opacity=.82)
        st.plotly_chart(style_plot(fig, 350), use_container_width=True)
    st.subheader("Predicted results")
    show_sent = st.selectbox("Filter by sentiment", ["All", "Positive", "Neutral", "Negative"])
    display = res if show_sent == "All" else res[res["Sentiment"] == show_sent]
    st.dataframe(display[["Review", "Compound score", "Sentiment"]], use_container_width=True, hide_index=True, height=350)
    if "label" in analysis_df.columns and set(pd.to_numeric(analysis_df["label"], errors="coerce").dropna().unique()).issubset({0,1}):
        st.subheader("Comparison with original Steam labels")
        labels = pd.to_numeric(analysis_df["label"], errors="coerce").map({1:"Positive", 0:"Negative"})
        compare = pd.crosstab(labels, res["Sentiment"]).reindex(index=["Positive","Negative"], columns=["Positive","Neutral","Negative"], fill_value=0)
        st.dataframe(compare, use_container_width=True)
        st.caption("Steam's original labels are binary thumbs-up/thumbs-down labels. They are not the same thing as the three-class predictions above.")
        if len(np.unique(pd.to_numeric(analysis_df["label"], errors="coerce").dropna())) == 2 and len(Z) > 5:
            try:
                y = pd.to_numeric(analysis_df["label"], errors="coerce").astype(int)
                Xtr, Xte, ytr, yte = train_test_split(Z, y, test_size=.2, random_state=42, stratify=y)
                clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
                with st.expander("Optional supervised check: Logistic Regression"):
                    st.code(classification_report(yte, clf.predict(Xte), zero_division=0))
                    st.caption("This optional check predicts the original binary Steam labels; it is separate from VADER's three-class sentiment output.")
            except Exception as exc:
                st.caption(f"The optional supervised comparison could not be calculated for this dataset: {exc}")
    st.download_button("Download sentiment results (CSV)", res.to_csv(index=False).encode("utf-8"), "pacify_sentiment_results.csv", "text/csv")

elif selected_page == "Tag Analysis":
    st.markdown("<div class='section-kicker'>STAGE 05</div>", unsafe_allow_html=True)
    st.header("Tag Generation & Categorization")
    st.markdown("<p class='section-intro'>Candidate terms are ranked, refined to reduce duplicates, then enriched with related words.</p>", unsafe_allow_html=True)
    tag_tabs = st.tabs(["Ranked candidates", "Refined tags", "Enriched tags & cloud"])
    with tag_tabs[0]:
        st.dataframe(ranked.head(top_k), use_container_width=True, hide_index=True, height=420)
        st.caption("Ranking score combines TF-IDF weight with the number of reviews containing the term.")
    with tag_tabs[1]:
        st.dataframe(refined[["Tag candidate", "Ranking score", "Documents present", "Dominant sentiment"]], use_container_width=True, hide_index=True, height=420)
        st.caption("Refinement filters candidates by part of speech where possible and merges stem variants.")
    with tag_tabs[2]:
        left, right = st.columns([1.15, .85], gap="large")
        with left:
            weights = {str(row["Tag candidate"]): float(row["Ranking score"]) for _, row in refined.iterrows() if float(row["Ranking score"]) > 0}
            for _, row in refined.iterrows():
                for related in row["Related tags"]:
                    weights.setdefault(str(related), float(row["Ranking score"]) * .3)
            if weights:
                cloud = WordCloud(width=1000, height=600, background_color="#FFFCF8", colormap="RdPu", prefer_horizontal=.88, max_words=100, collocations=False, random_state=42).generate_from_frequencies(weights)
                fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="#FFFCF8")
                ax.imshow(cloud, interpolation="bilinear")
                ax.axis("off")
                fig.tight_layout(pad=0)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.info("No tags were generated from the current text.")
        with right:
            st.dataframe(refined[["Tag candidate", "Related tags", "Dominant sentiment"]], use_container_width=True, hide_index=True, height=440)
        st.download_button("Download tag results (CSV)", refined.to_csv(index=False).encode("utf-8"), "pacify_tag_results.csv", "text/csv")

elif selected_page == "Review Explorer":
    st.markdown("<div class='section-kicker'>CORPUS EXPLORER</div>", unsafe_allow_html=True)
    st.header("Review Explorer")
    st.markdown("<p class='section-intro'>Search individual reviews and inspect their cleaned text, predicted sentiment and score.</p>", unsafe_allow_html=True)
    search = st.text_input("Search reviews", placeholder="Try: scary, bugs, multiplayer…")
    filter_sent = st.selectbox("Sentiment", ["All", "Positive", "Neutral", "Negative"])
    explorer = res.copy()
    if search.strip():
        explorer = explorer[explorer["Review"].str.contains(search.strip(), case=False, regex=False)]
    if filter_sent != "All":
        explorer = explorer[explorer["Sentiment"] == filter_sent]
    st.caption(f"Showing {len(explorer):,} of {review_count:,} reviews")
    st.dataframe(explorer[["Review", "Cleaned text", "Compound score", "Sentiment"]], use_container_width=True, hide_index=True, height=500)
    st.download_button("Download analysed reviews (CSV)", res.to_csv(index=False).encode("utf-8"), "pacify_analysed_reviews.csv", "text/csv")

st.markdown("---")
