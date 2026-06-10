import streamlit as st
from src.preprocessing import load_data
from src.recommender import MovieRecommender


st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("🎬 Movie Recommendation System")
st.write("Get personalized movie recommendations based on your ratings")

# -------------------------
# Load data
# -------------------------
movies, ratings = load_data()

recommender = MovieRecommender(movies, ratings)

# -------------------------
# Sample movies for rating
# -------------------------
sample_movies = ["Toy Story", "Jumanji", "Pulp Fiction", "Akira"]

st.header("⭐ Rate Movies")

user_ratings = {}

for movie in sample_movies:
    user_ratings[movie] = st.slider(movie, 0.0, 5.0, 3.0, 0.5)

# -------------------------
# Convert input to dataframe format
# -------------------------
input_movies = movies[movies["title"].isin(user_ratings.keys())].copy()

input_movies["rating"] = input_movies["title"].map(user_ratings)

# -------------------------
# Button
# -------------------------
if st.button("🎯 Get Recommendations"):

    recommender.build_user_subset(input_movies)
    recommender.compute_similarity(input_movies)

    recommendations = recommender.recommend(input_movies)

    st.subheader("🔥 Top Recommendations")

    for _, row in recommendations.head(10).iterrows():
        st.write(f"🎬 {row['title']} ({row.get('year', '')})")
