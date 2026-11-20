import streamlit as st
import pandas as pd
from recommender import NetflixRecommender

st.set_page_config(page_title='Netflix Recommendation System', page_icon='🎬', layout='wide')
st.title('🎬 Netflix Content Recommendation System')
st.caption('Content-based recommendations using TF-IDF and cosine similarity. Not affiliated with Netflix.')

@st.cache_resource
def get_engine():
    return NetflixRecommender()

engine = get_engine()
df = engine.df

with st.sidebar:
    st.header('Filters')
    category = st.selectbox('Content type', ['All', 'Movie', 'TV Show'])
    top_n = st.slider('Number of recommendations', 3, 15, 5)
    st.info('Similarity is based on genres, director, country, content type and rating — not user watch history.')

choices = df if category == 'All' else df[df['type'] == category]
# Include type in label to disambiguate duplicate names.
labels = [f'{row.title} ({row.type})' for row in choices.itertuples()]
selection = st.selectbox('Choose a Netflix title', range(len(labels)), format_func=lambda i: labels[i])
selected = choices.iloc[selection]
st.write(f"**Selected:** {selected['title']} — {selected['listed_in']}")

if st.button('Find similar titles', type='primary'):
    result = engine.recommend(selected['title'], top_n, content_type=selected['type'])
    st.subheader(f'Top {len(result)} recommendations')
    for i, item in result.iterrows():
        with st.container(border=True):
            left, right = st.columns([4, 1])
            left.markdown(f"**{i+1}. {item['title']}**")
            left.caption(f"{item['type']} · {item['release_year']} · {item['listed_in']}")
            left.write(f"Director: {item['director']} | Country: {item['country']}")
            right.metric('Similarity', f"{item['similarity_score']:.3f}")
            right.caption(f"Genre overlap: {item['genre_overlap']:.3f}")
    st.download_button('Download recommendations (CSV)', result.to_csv(index=False),
                       file_name='netflix_recommendations.csv', mime='text/csv')

with st.expander('About the model'):
    st.write('TF-IDF converts combined content attributes into sparse numerical vectors. Cosine similarity ranks the closest titles. Genre overlap is a proxy for relevance, not ground-truth accuracy.')
