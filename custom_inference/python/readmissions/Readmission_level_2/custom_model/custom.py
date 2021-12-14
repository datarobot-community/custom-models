import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder

import os
import io
from typing import List, Optional
from scipy.special import expit
g_code_dir = None

schema = {"race": "object", "gender": "object", "age": "object", "weight": "object", "admission_type_id": "object", "discharge_disposition_id": "object", "admission_source_id": "object", "time_in_hospital": "int64", "payer_code": "object", "medical_specialty": "object", "num_lab_procedures": "int64", "num_procedures": "int64", "num_medications": "int64", "number_outpatient": "int64", "number_emergency": "int64", "number_inpatient": "int64", "number_diagnoses": "int64", "max_glu_serum": "object", "A1Cresult": "object", "metformin": "object", "repaglinide": "object", "nateglinide": "object", "chlorpropamide": "object", "glimepiride": "object", "acetohexamide": "object", "glipizide": "object", "glyburide": "object", "tolbutamide": "object", "pioglitazone": "object", "rosiglitazone": "object", "acarbose": "object", "miglitol": "object", "troglitazone": "object", "tolazamide": "object", "examide": "object", "citoglipton": "object", "insulin": "object", "glyburide_metformin": "object", "glipizide_metformin": "object", "glimepiride_pioglitazone": "object", "metformin_rosiglitazone": "object", "metformin_pioglitazone": "object", "change": "object", "diabetesMed": "object"}

def init(code_dir):
    global g_code_dir
    g_code_dir = code_dir

def read_input_data(input_binary_data):
    data = pd.read_csv(io.BytesIO(input_binary_data))
    data.drop(['diag_1_desc', 'diag_1', 'diag_2', 'diag_3'],axis=1,inplace=True)

    #Saving this for later
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

    #Handle null values in categories and numerics
    for c,dt in schema.items():
        if dt =='object':
            data[c] = data[c].fillna('missing')
        else:
            data[c] = data[c].fillna(0)

    pipeline_path = 'preprocessing.pkl'
    pipeline = joblib.load(os.path.join(g_code_dir, pipeline_path))
    preprocessed = pipeline.transform(data)
    preprocessed = pd.DataFrame.sparse.from_spmatrix(preprocessed)
    
    return preprocessed

def load_model(code_dir):
    model_path = 'model.pkl'
    model = joblib.load(os.path.join(code_dir, model_path))
    return model

def score(data, model, **kwargs):
    results = model.predict_proba(data)
    predictions = pd.DataFrame({'True': results[:, 0], 'False':results[:, 1]})

    return predictions
