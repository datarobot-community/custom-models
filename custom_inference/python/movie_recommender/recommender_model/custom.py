import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
import os
import io
from io import StringIO


def load_model(code_dir):
    model_path = 'saved_model.pb'
    model = tf.keras.models.load_model('./recommender_model/',
                                       custom_objects = None, 
                                       compile = True, 
                                       options = None)
    print ("model loaded")
    return model


def score(data, model, **kwargs):
    predictions = model.predict(data)
    print (predictions)
    s = pd.DataFrame(predictions)
    return s


def score_unstructured(model, data, query, **kwargs):

    """
    Load Movie/User Info
    
    This will be used to output the Actual movie list
    instead of just the indexes of the movies
    """
    
    movie_df = pd.read_csv('movies.csv')
    df = pd.read_csv('ratings_file.csv')
    data_rec = pd.read_csv('predict.csv')
    
    user_id = data_rec.userID.iloc[0]   
    movies_watched_by_user = df[df.userId == user_id]
   
    
    # Find Movies Not Watched
    movies_not_watched = movie_df[
        ~movie_df["movieId"].isin(movies_watched_by_user.movieId.values)
    ]["movieId"]

    user_ids = df["userId"].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    userencoded2user  = {i: x for i, x in enumerate(user_ids)}
    movie_ids = df["movieId"].unique().tolist()
    movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
    movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}

    movies_not_watched = list(
        set(movies_not_watched).intersection(set(movie2movie_encoded.keys()))
    )

    movies_not_watched = [[movie2movie_encoded.get(x)] for x in movies_not_watched]
    user_encoder = user2user_encoded.get(user_id)
    user_movie_array = np.hstack(
        ([[user_encoder]] * len(movies_not_watched), movies_not_watched)
    )
    
    """
    Finished Loading Movie data
    """
    
    input = io.StringIO(data) 
    X = pd.read_csv(input)  
    
    # Fill NA
    # Cast Inputs as INTs to properly handle NULL value imputation
    # and prevent from being cast as Floats
    X['userID'].fillna(user_id, inplace=True)
    X['movies'].fillna(3678, inplace=True)
    X["userID"] = X["userID"].astype(int) 
    X["movies"] = X["movies"].astype(int)
    

    ratings = model.predict(X).flatten() 
    
    # Take the Top 10 Movie Recommendations
    top_ratings_indices = ratings.argsort()[-10:][::-1]
    recommended_movie_ids = [
        movie_encoded2movie.get(movies_not_watched[x][0]) for x in top_ratings_indices
    ]
    
    recommended_movies = movie_df[movie_df["movieId"].isin(recommended_movie_ids)]
    for row in recommended_movies.itertuples():
        print(row.title, ":", row.genres)

#     print (recommended_movies)
    s = pd.DataFrame(recommended_movies)
    t = s.to_csv(index=False)
    return t


