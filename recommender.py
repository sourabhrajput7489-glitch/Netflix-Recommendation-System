"""Content-based Netflix recommendation engine using TF-IDF and cosine similarity."""
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path(__file__).resolve().parent / 'data' / 'Dataset.csv'


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    required = ['title', 'type', 'director', 'country', 'rating', 'listed_in']
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    df = df.drop_duplicates(subset=['title', 'type']).reset_index(drop=True).copy()
    df[required] = df[required].fillna('').astype(str)
    return df


def clean_tokens(text):
    return re.sub(r'[^a-z0-9 ]+', ' ', str(text).lower()).strip()


def build_features(df):
    df = df.copy()
    # Duplicate genres to give category matches higher relative weight.
    df['combined_features'] = (
        df['listed_in'].map(clean_tokens) + ' ' +
        df['listed_in'].map(clean_tokens) + ' ' +
        df['director'].map(clean_tokens) + ' ' +
        df['country'].map(clean_tokens) + ' ' +
        df['type'].map(clean_tokens) + ' ' +
        df['rating'].map(clean_tokens)
    )
    return df


class NetflixRecommender:
    def __init__(self, path=DATA_PATH):
        self.df = build_features(load_data(path))
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(self.df['combined_features'])

    def recommend(self, title, top_n=5, content_type=None):
        matches = self.df.index[self.df['title'].str.casefold() == title.strip().casefold()].tolist()
        if content_type:
            matches = [i for i in matches if self.df.at[i, 'type'] == content_type]
        if not matches:
            raise ValueError(f'Title not found: {title}')
        idx = matches[0]
        scores = cosine_similarity(self.matrix[idx], self.matrix).ravel()
        scores[idx] = -1
        order = np.argsort(-scores, kind='stable')[:min(max(1, int(top_n)), len(scores)-1)]
        cols = ['title', 'type', 'listed_in', 'director', 'country', 'release_year']
        result = self.df.iloc[order][cols].copy()
        result['similarity_score'] = np.round(scores[order], 3)
        result['genre_overlap'] = [self.genre_overlap(self.df.at[idx, 'listed_in'], s) for s in result['listed_in']]
        return result.reset_index(drop=True)

    @staticmethod
    def genre_overlap(a, b):
        a = {x.strip().casefold() for x in a.split(',') if x.strip()}
        b = {x.strip().casefold() for x in b.split(',') if x.strip()}
        return round(len(a & b) / len(a | b), 3) if a | b else 0.0

    def evaluate(self, titles, top_n=5):
        rows = []
        for title in titles:
            try:
                recs = self.recommend(title, top_n)
                rows.append({'title': title, 'mean_genre_overlap': recs['genre_overlap'].mean(),
                             'mean_cosine_similarity': recs['similarity_score'].mean()})
            except ValueError:
                continue
        return pd.DataFrame(rows)


if __name__ == '__main__':
    engine = NetflixRecommender()
    print(f'Loaded {len(engine.df)} unique title/type records')
    print(engine.recommend('Ganglands').to_string(index=False))
