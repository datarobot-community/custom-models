from tensorflow import keras
import os
import pandas as pd
import joblib
import scipy.sparse

g_code_dir = None
def init(code_dir):
    global g_code_dir
    g_code_dir = code_dir

def transform(data, model):
    """
    Note: This hook may not have to be implemented for your model.
    In this case implemented for the model used in the example.
    Modify this method to add data transformation before scoring calls. For example, this can be
    used to implement one-hot encoding for models that don't include it on their own.
    Parameters
    ----------
    data: pd.DataFrame
    model: object, the deserialized model
    Returns
    -------
    pd.DataFrame
    """
    # Execute any steps you need to do before scoring

    pipeline_path = 'preprocessing.pkl'
    pipeline = joblib.load(os.path.join(g_code_dir, pipeline_path))
    transformed = pipeline.transform(data)
    data = pd.DataFrame.sparse.from_spmatrix(transformed)
    
    return data

def load_model(code_dir):
    model_path = 'model.h5'
    model = keras.models.load_model(os.path.join(code_dir, model_path))
    return model

def score(data, model, **kwargs):
    results = model.predict_proba(data)

    #Create two columns with probability results
    predictions = pd.DataFrame({'yes': results[:, 0]})
    predictions['no'] = 1 - predictions['yes']

    return predictions
