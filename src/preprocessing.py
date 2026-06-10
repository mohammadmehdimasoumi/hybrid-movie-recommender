import pandas as pd


def load_data(movies_path="data/movies.csv", ratings_path="data/ratings.csv"):
    """
    Load and clean MovieLens dataset
    """

    # Load datasets
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)

    # ----------------------------
    # Clean movies dataset
    # ----------------------------

    # Extract year from title
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)", expand=False)

    # Remove year from title
    movies["title"] = movies["title"].str.replace(r"\(\d{4}\)", "", regex=True)

    # Strip spaces
    movies["title"] = movies["title"].str.strip()

    # Drop genres column (optional)
    if "genres" in movies.columns:
        movies = movies.drop("genres", axis=1)

    # ----------------------------
    # Clean ratings dataset
    # ----------------------------

    if "timestamp" in ratings.columns:
        ratings = ratings.drop("timestamp", axis=1)

    # ----------------------------
    # Feature engineering
    # ----------------------------

    if "genres" in movies.columns:
        movies["content"] = movies["title"].fillna("") + " " + movies["genres"].fillna("")
    else:
        movies["content"] = movies["title"].fillna("")

    return movies, ratings


# ----------------------------
# Test run
# ----------------------------
if __name__ == "__main__":
    movies, ratings = load_data()
    print("Movies shape:", movies.shape)
    print("Ratings shape:", ratings.shape)
    print(movies.head())


