# Recommender System Unstructured Example

This folder includes an example of a Keras movie recommender system. The model was built using the Notebook [here](https://keras.io/examples/structured_data/collaborative_filtering_movielens/)

Using the saved model from that notebook, we then use DRUM to validate and then score using the saved model. 

In the `custom.py` script, we use the hook functions including the `load_model` and `score_unstructured` functions. 

Additional modifications have been made to the `score_unstructured` function to output the actual names of the movies
as opposed to the index of the movieId. 

