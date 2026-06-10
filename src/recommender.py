import pandas as pd
import numpy as np
from math import sqrt


class MovieRecommender:
    def __init__(self, movies_df, ratings_df):
        self.movies = movies_df
        self.ratings = ratings_df
        self.user_subset_group = None
        self.pearson_dict = {}

    # -------------------------
    # Prepare user subset
    # -------------------------
    def build_user_subset(self, input_movies):
        """
        Filter users who watched same movies as input user
        """

        user_subset = self.ratings[
            self.ratings["movieId"].isin(input_movies["movieId"].tolist())
        ]

        self.user_subset_group = user_subset.groupby("userId")

    # -------------------------
    # Compute Pearson similarity
    # -------------------------
    def compute_similarity(self, input_movies, top_n=50):
        """
        Compute Pearson correlation between input user and other users
        """

        self.user_subset_group = sorted(
            self.user_subset_group,
            key=lambda x: len(x[1]),
            reverse=True
        )

        self.user_subset_group = self.user_subset_group[:top_n]

        self.pearson_dict = {}

        for name, group in self.user_subset_group:

            group = group.sort_values(by="movieId")
            input_movies = input_movies.sort_values(by="movieId")

            n_ratings = len(group)

            temp_df = input_movies[
                input_movies["movieId"].isin(group["movieId"].tolist())
            ]

            if len(temp_df) == 0:
                continue

            temp_rating_list = temp_df["rating"].tolist()
            temp_group_list = group["rating"].tolist()

            Sxx = sum([i**2 for i in temp_rating_list]) - (sum(temp_rating_list) ** 2) / float(n_ratings)
            Syy = sum([i**2 for i in temp_group_list]) - (sum(temp_group_list) ** 2) / float(n_ratings)
            Sxy = sum(i * j for i, j in zip(temp_rating_list, temp_group_list)) - (sum(temp_rating_list) * sum(temp_group_list)) / float(n_ratings)

            if Sxx != 0 and Syy != 0:
                self.pearson_dict[name] = Sxy / sqrt(Sxx * Syy)
            else:
                self.pearson_dict[name] = 0

    # -------------------------
    # Generate recommendations
    # -------------------------
    def recommend(self, input_movies, top_k=10):
        """
        Return top recommended movies
        """

        pearson_df = pd.DataFrame.from_dict(
            self.pearson_dict,
            orient="index",
            columns=["similarity"]
        )

        pearson_df["userId"] = pearson_df.index

        top_users = pearson_df.sort_values(
            by="similarity",
            ascending=False
        ).head(50)

        top_users_ratings = top_users.merge(
            self.ratings,
            on="userId"
        )

        # Weighted rating
        top_users_ratings["weightedRating"] = (
            top_users_ratings["similarity"] * top_users_ratings["rating"]
        )

        temp = top_users_ratings.groupby("movieId").sum()[["similarity", "weightedRating"]]

        temp.columns = ["sum_similarity", "sum_weightedRating"]

        recommendation_df = pd.DataFrame()

        recommendation_df["score"] = (
            temp["sum_weightedRating"] / temp["sum_similarity"]
        )

        recommendation_df["movieId"] = temp.index

        recommendation_df = recommendation_df.sort_values(
            by="score",
            ascending=False
        )

        # Return final movies
        return self.movies[
            self.movies["movieId"].isin(recommendation_df.head(top_k)["movieId"])
        ]
