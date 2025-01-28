import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import sys


def recommend_movies_with_threshold(ratings_file, similarity_threshold=None):
    """
    Recommends movies based on user ratings and an optional similarity threshold.

    Args:
        ratings_file: Path to the CSV file containing user-movie ratings.
        similarity_threshold: The minimum similarity score for a recommendation to be considered.

    Returns:
        A list of tuples (user_id, movie_id, predicted_rating).
    """
    df = pd.read_csv(ratings_file)

    # Create user-item matrix
    user_item_matrix = df.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)

    # Calculate cosine similarity between movies
    movie_similarity = cosine_similarity(user_item_matrix.T)
    movie_similarity_df = pd.DataFrame(movie_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

    def predict_rating(user_id, movie_id):
        """Predicts a user's rating for a movie using item-based collaborative filtering."""
        user_ratings = user_item_matrix.loc[user_id]
        movie_similarities = movie_similarity_df[movie_id]

        # Weighted average of ratings for similar movies
        predicted_rating = np.dot(user_ratings, movie_similarities) / np.sum(np.abs(movie_similarities))

        return predicted_rating

    def recommend_movies(user_id, num_recommendations=5):
        """Recommends movies for a user based on predicted ratings."""
        unrated_movies = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] == 0].index.tolist()
        predicted_ratings = {}
        for movie_id in unrated_movies:
            predicted_ratings[movie_id] = predict_rating(user_id, movie_id)

        # Sort movies by predicted rating in descending order
        sorted_predictions = dict(sorted(predicted_ratings.items(), key=lambda item: item[1], reverse=True))

        # Return the top N recommended movies
        return list(sorted_predictions.keys())[:num_recommendations]


    recommendations = []
    for user_id in user_item_matrix.index:
        for movie_id in recommend_movies(user_id):
            predicted_rating = predict_rating(user_id, movie_id)
            if similarity_threshold is None or predicted_rating >= similarity_threshold:
                recommendations.append((user_id, movie_id, predicted_rating))

    return recommendations

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python your_program_name.py <ratings_file> [similarity_threshold]")
        sys.exit(1)

    ratings_file = sys.argv[1]
    similarity_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else None

    recommendations = recommend_movies_with_threshold(ratings_file, similarity_threshold)

    for user_id, movie_id, predicted_rating in recommendations:
        print(f"{user_id} {movie_id} {predicted_rating}") 