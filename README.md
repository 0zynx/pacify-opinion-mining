# Pacify Review Analysis

A text mining project analysing Steam reviews of *Pacify*, a multiplayer horror game, using TF-IDF, opinion classification, dimensionality reduction, and tag generation to uncover player sentiment and recurring themes.

## Features

- **Text Preprocessing:** Cleaning, tokenization, stop-word removal, stemming, and lemmatization.
- **Bag-of-Words:** Represents reviews using word frequencies.
- **TF-IDF:** Identifies important words and phrases in reviews.
- **Dimensionality Reduction:** Uses SVD/LSA to reduce textual features and visualize review patterns.
- **Opinion Classification:** Categorizes reviews as positive, negative, or neutral.
- **Tag Generation:** Ranks, refines, and enriches tags to identify recurring topics.
- **Review Explorer:** Allows individual reviews and their analysis results to be explored.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- NLTK
- Streamlit
- Plotly and Matplotlib
- WordCloud

## Dataset

The project uses player reviews of *Pacify* collected from Steam.

- **Game:** Pacify
- **Platform:** Steam
- **Steam App ID:** 967050

[View Pacify on Steam](https://store.steampowered.com/app/967050/Pacify/)

## Project Structure

```text
pacify-opinion-mining/
├── app.py
├── collect_reviews.py
├── clean_reviews.py
├── requirements.txt
├── reviews.csv
├── reviews_clean.csv
└── .gitignore
```

## Installation and Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/0zynx/pacify-opinion-mining.git
   cd pacify-opinion-mining
   ```

2. Create and activate a virtual environment (optional but recommended).

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:

   ```bash
   python -m streamlit run app.py
   ```

5. Open the local URL displayed in the terminal to explore the dashboard.

## Data Collection

To collect reviews from Steam, run:

```bash
python collect_reviews.py
```

To clean the collected reviews, run:

```bash
python clean_reviews.py
```

The scripts generate the CSV files used by the project. Review counts may vary depending on the collected data.

## Project Goal

The goal is to transform unstructured game reviews into useful insights by identifying important terms, analysing player opinions, and discovering recurring themes through text mining.
