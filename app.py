from src.preprocessing import load_data
from src.recommender import MovieRecommender


def main():

    # Load data
    movies, ratings = load_data()

    # Create recommender system
    recommender = MovieRecommender(movies, ratings)

    # -----------------------------
    # User input (simulated user)
    # -----------------------------
    user_input = [
        {"title": "Toy Story", "rating": 3.5},
        {"title": "Jumanji", "rating": 2},
        {"title": "Pulp Fiction", "rating": 5},
        {"title": "Akira", "rating": 4.5}
    ]

    input_movies = movies[movies["title"].isin([m["title"] for m in user_input])].copy()

    # add ratings
    input_movies["rating"] = input_movies["title"].map(
        {m["title"]: m["rating"] for m in user_input}
    )

    # -----------------------------
    # Run recommender
    # -----------------------------
    recommender.build_user_subset(input_movies)
    recommender.compute_similarity(input_movies)
    recommendations = recommender.recommend(input_movies)

    # -----------------------------
    # Output
    # -----------------------------
    print("\n🔥 Top Movie Recommendations:\n")
    print(recommendations[["title", "year"]].head(10))


if __name__ == "__main__":
    main()
