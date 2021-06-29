import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
import os
import io
from io import StringIO


def load_model(code_dir):
    model_path = 'mnist.h5'
    print ("model ready to load")
    model = keras.models.load_model(os.path.join(code_dir, model_path), compile=False)
    print ("model loaded")
    return model

def score(data, model, **kwargs):
    print (data.shape)
    X=data.drop(data.columns[0],axis=1)
    X=X.values.reshape(X.shape[0],28,28,1)
    predictions = model.predict(X)
    print (predictions)
    s = pd.DataFrame(predictions)
    return s

def score_unstructured(model, data, query, **kwargs):
    print("Incoming content type params: ", kwargs)
    print("Incoming data type: ", type(data))
    print("Incoming query params: ", query)
    input = io.StringIO(data)
    X = pd.read_csv(input)
    print (X.shape)
    X=X.drop(X.columns[0],axis=1)
    X=X.values.reshape(X.shape[0],28,28,1)
    predictions = model.predict(X)
    print (predictions)
    s = pd.DataFrame(predictions)
    t = s.to_csv(index=False)
    return t


