import pandas as pd
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process

# Read and process ratings
rating = pd.read_csv('../data/ratings.csv')
df_rating = pd.DataFrame(rating)
df_rating = df_rating.drop('timestamp', axis=1)
print(df_rating.head())

# Read and process movie metadata
movie_metadata = pd.read_csv('../data/movies.csv', low_memory=False)
movie_metadata = movie_metadata[['title', 'genres', 'movieId']]
print(movie_metadata.head())

# Merge ratings with movie metadata
movie_data = df_rating.merge(movie_metadata, on='movieId')
print(movie_data.head())

# Create user-item matrix
user_item_matrix = df_rating.pivot(index='userId', columns='movieId', values='rating').fillna(0)
print(user_item_matrix)

# Define KNN model using cosine similarity
knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10, n_jobs=1)
knn_model.fit(user_item_matrix)

# Recommendation system function
def movie_recommendation_system(movie_name, matrix, cf_model, n_recs):
    # Transpose matrix for item-based filtering
    cf_model.fit(matrix.T)

    # Fuzzy match movie title
    title_list = movie_metadata['title'].tolist()
    matched_title = process.extractOne(movie_name, title_list)[0]
    movie_id = movie_metadata[movie_metadata['title'] == matched_title]['movieId'].values[0]

    # Get the vector for the selected movie
    movie_vector = matrix.T.loc[movie_id].values.reshape(1, -1)

    # Find nearest neighbors
    distances, indices = cf_model.kneighbors(movie_vector, n_neighbors=n_recs + 1)

    # Store recommendations (skip the first one, which is the movie itself)
    cf_recs = []
    for i in range(1, n_recs + 1):
        rec_movie_id = matrix.T.index[indices[0][i]]
        rec_title = movie_metadata[movie_metadata['movieId'] == rec_movie_id]['title'].values[0]
        cf_recs.append({'Title': rec_title, 'Distance': round(distances[0][i], 4)})

    # Return as DataFrame
    df = pd.DataFrame(cf_recs, index=range(1, n_recs + 1))
    return df

movie_name = input("Enter a Movie Name of your Taste for Recommendation: ")
recommendations = movie_recommendation_system(movie_name, user_item_matrix, knn_model, 10)
print(recommendations)