# Netflix Content Recommendation System

Content-based machine learning internship project using 8,790 Netflix records (`data/Dataset.csv`).

## Goal
Recommend similar Netflix titles based on genre/category, director, country, type and content rating.

## Tech stack
Python, Pandas, NumPy, Scikit-learn (TF-IDF and cosine similarity), Matplotlib, Jupyter Notebook, Streamlit.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
- Notebook: `jupyter notebook notebooks/netflix_recommendation.ipynb`
- CLI: `python recommender.py`
- Web app: `streamlit run app.py`

## Approach
1. Inspect and clean data; remove 3 duplicate title/type records.
2. Combine and normalize content metadata, weighting genres twice.
3. Encode text with TF-IDF unigrams and bigrams.
4. Compute on-demand cosine similarity for selected title.
5. Display ranked titles and assess genre overlap (Jaccard index).

## Evaluation limitation
Genre overlap and cosine similarity are **proxy relevance indicators**, not prediction accuracy. Actual recommendation effectiveness requires user feedback, clicks, ratings, or labeled relevance judgments.

## Files
- `data/Dataset.csv` — original supplied dataset
- `notebooks/netflix_recommendation.ipynb` — exploration, model, evaluation, chart
- `recommender.py` — reusable model
- `app.py` — Streamlit application

## Note
This educational project is not affiliated with Netflix. It uses only metadata; no posters or watch-history personalization are included.
