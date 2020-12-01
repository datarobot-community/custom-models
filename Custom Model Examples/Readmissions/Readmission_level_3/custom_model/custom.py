from tensorflow import keras
import os
import pandas as pd
import joblib
import scipy.sparse
from scipy.special import expit

g_input_filename = None
g_code_dir = None

def init(code_dir):
    global g_code_dir
    g_code_dir = code_dir

def read_input_data(input_filename):
    data = pd.read_csv(input_filename)

    #Saving this for later
    global g_input_filename
    g_input_filename = input_filename
    return data

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
    results = model.predict(data)

    #Create two columns with probability results
    predictions = pd.DataFrame({'yes': results[:, 0]})
    predictions['no'] = 1 - predictions['yes']

    return predictions

#Adding post_process to use legacy model together with Keras model
def post_process(predictions,model):
    original_data = pd.read_csv(g_input_filename)
    original_data.fillna(0,inplace=True)

    def legacy_score(row):
        try:
            return expit(0.59 + 0.55 * row['number_inpatient'] + 0.36 * row['number_outpatient'])
        except:
            return 0.38

    predictions['yes_legacy'] = original_data.apply(lambda row: legacy_score(row), axis=1)
    predictions['yes'] = (predictions['yes'] + predictions['yes_legacy'])
    predictions['yes'] = predictions['yes']/2
    predictions['no'] = 1 -  predictions['yes']

    predictions.drop('yes_legacy',axis=1,inplace=True)

    return predictions