#MOVIE RECOMMANDATION SYSTEM

# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

#Loading Rating Dataset
ratings = pd.read_csv("https://s3-us-west-2.amazonaws.com/recommender-tutorial/ratings.csv")
print(ratings.rename(columns={'userId': 'USERID', 'movieId': 'MOVIEID', 'rating': 'RATING', 'timestamp': 'TIMESTAMP'}).head())

#Loading Movie Dataset
movies = pd.read_csv("https://s3-us-west-2.amazonaws.com/recommender-tutorial/movies.csv")
print(movies.rename(columns={'movieId': 'MOVIEID', 'title': 'TITLE', 'genres': 'GENRES'}).head())

n_ratings = len(ratings)
n_movies = len(ratings['movieId'].unique())
n_users = len(ratings['userId'].unique())

print(f"\nNumber of ratings: {n_ratings}")
print(f"Number of unique movieId's: {n_movies}")
print(f"Number of unique users: {n_users}")
print(f"Average ratings per user: {round(n_ratings / n_users, 2)}")
print(f"Average ratings per movie: {round(n_ratings / n_movies, 2)}")

user_freq = ratings[['userId', 'movieId']].groupby('userId').count().reset_index()
user_freq.columns = ['USERID', 'N_RATINGS']
print("\nUser Ratings Count:")
print(user_freq.head())

# Find Lowest and Highest Rated Movies:
mean_rating = ratings.groupby('movieId')[['rating']].mean()

# Lowest rated movies
lowest_rated = mean_rating['rating'].idxmin()
print("\nLowest Rated Movie:")
print(movies.loc[movies['movieId'] == lowest_rated])

# Highest rated movies
highest_rated = mean_rating['rating'].idxmax()
print("\nHighest Rated Movie:")
print(movies.loc[movies['movieId'] == highest_rated])

# Show number of people who rated movies rated highest
print("\nPeople who Rated Highest Rated Movie:")
print(ratings[ratings['movieId'] == highest_rated])

# Show number of people who rated movies rated lowest
print("\nPeople who Rated Lowest Rated Movie:")
print(ratings[ratings['movieId'] == lowest_rated])

# The above movies have very low dataset. We will use Bayesian average
movie_stats = ratings.groupby('movieId')[['rating']].agg(['count', 'mean'])
movie_stats.columns = movie_stats.columns.droplevel()
