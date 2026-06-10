import pandas as pd
import numpy as np


class Evaluator:
    def __init__(self, movies_df, ratings_df):
        self.movies = movies_df
        self.ratings = ratings_df

    # -------------------------
    # Hit Rate @ K
    # -------------------------
    def hit_rate_at_k(self, recommended_movies, user_liked_movies, k=10):
        """
        Checks how many recommended movies exist in user liked movies
        """

        recommended_top_k = recommended_movies.head(k)["movieId"].tolist()

        liked_set = set(user_liked_movies["movieId"].tolist())

        hits = 0

        for movie in recommended_top_k:
            if movie in liked_set:
                hits += 1

        return hits / k if k > 0 else 0


    # -------------------------
    # Simple evaluation report
    # -------------------------
    def evaluate(self, recommended_movies, user_liked_movies):
        """
        Print basic evaluation metrics
        """

        hit_rate = self.hit_rate_at_k(
            recommended_movies,
            user_liked_movies,
            k=10
        )

        print("\n📊 Evaluation Report")
        print("----------------------")
        print(f"Hit Rate @10: {hit_rate:.2f}")
