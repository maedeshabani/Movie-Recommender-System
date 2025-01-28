
import numpy as np
import pandas as pd
from recommender.py import recommend_movies_with_threshold

def evaluate_recommender(test_df, recommendations):
  """Evaluates the recommender system using RMSE, precision, and recall.

  Args:
    test_df: DataFrame containing user-movie ratings for the test set.
    recommendations: List of tuples (user_id, movie_id, predicted_rating).

  Returns:
    A dictionary containing the RMSE, precision, and recall scores.
  """
  true_ratings = []
  predicted_ratings = []

  for user_id, movie_id, predicted_rating in recommendations:
    actual_rating = test_df[(test_df['user_id'] == user_id) & (test_df['movie_id'] == movie_id)]['rating']
    if not actual_rating.empty:
        true_ratings.append(actual_rating.iloc[0])
        predicted_ratings.append(predicted_rating)

 # Calculate RMSE
  rmse = np.sqrt(np.mean((np.array(true_ratings) - np.array(predicted_ratings))**2))

  # Calculate precision and recall
  ''' in order to filter relevant recommendation for true and false label classification,
   we determine a threshold = 3.5 for predicted_ratings.'''
  
  relevant_recommendations = [(user_id, movie_id) for user_id, movie_id, predicted_rating in recommendations if predicted_rating >= 3.5]
  true_positives = 0
  false_positives = 0
  false_negatives = 0

  for user_id, movie_id in relevant_recommendations:
    actual_rating = test_df[(test_df['user_id'] == user_id) & (test_df['movie_id'] == movie_id)]['rating']
    if not actual_rating.empty and actual_rating.iloc[0] >= 3.5:
      true_positives += 1
    elif not actual_rating.empty and actual_rating.iloc[0] < 3.5:
      false_positives += 1
    else:
      false_positives +=1
    
  # Iterate through the test set to count true negatives and false negatives
  for user_id in test_df['user_id'].unique():
    for movie_id in test_df['movie_id'].unique():
        if (user_id, movie_id) not in relevant_recommendations:
            actual_rating = test_df[(test_df['user_id'] == user_id) & (test_df['movie_id'] == movie_id)]['rating']
            if not actual_rating.empty and actual_rating.iloc[0] >= 3.5:
                false_negatives += 1
  if true_positives + false_positives == 0:
    precision = 0
  else:
    precision = true_positives / (true_positives + false_positives)

  if true_positives + false_negatives == 0:
    recall = 0
  else:
    recall = true_positives / (true_positives + false_negatives)

  return {'rmse': rmse, 'precision': precision, 'recall': recall}

# Load the test data
test_df = pd.read_csv('test_data.csv')

# Assume 'recommendations' is a list of (user_id, movie_id, predicted_rating) tuples
recommendations = recommend_movies_with_threshold('test_data.csv')

# Evaluate the recommender
evaluation_results = evaluate_recommender(test_df, recommendations)
evaluation_results