import pickle
import os
import pandas as pd
import numpy as np
from typing import List, Optional
from catboost import CatBoostClassifier
import io

def read_input_data(input_binary_data):
    return pd.read_csv(io.BytesIO(input_binary_data))


def transform(data,model):
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

    def find_diabetes_text(txt):
        try:
            if 'diabetes' in txt.lower():
                return 1
            else:
                return 0
        except:
            0
            
    #DataRobot Drum will check what happens when values are imputed. That is why I explicetely define cat_features
    cat_features = ['race', 'gender', 'age', 'weight', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id', 'payer_code', 'medical_specialty', 'diag_1',
    'diag_2', 'diag_3', 'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
    'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
    'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin', 
    'glyburide_metformin', 'glipizide_metformin', 'glimepiride_pioglitazone', 'metformin_rosiglitazone', 
    'metformin_pioglitazone', 'change', 'diabetesMed']

    # Fill null values for Categorical Features
    for c in cat_features:
        data[c] = data[c].fillna('unknown')

        #Some categorical features (diag_1), have float values which in reality are categories. Catboost takes either int or object as input
        #so I am casting.
        try:
            data[c] = data[c].astype(int)
        except:
            pass

    # Find out if `Diabetes|`diabetes` exists in diag_1_desc column
    data['diabetes'] = data['diag_1_desc'].apply(lambda x: find_diabetes_text(x))
    data.drop('diag_1_desc',axis=1,inplace=True)

    # Fill null values for numeric features
    data = data.fillna(0)

    return data

def score(data, model, **kwargs):

    results = model.predict_proba(data)

    #Create two columns with probability results
    predictions = pd.DataFrame({'True': results[:, 0]})
    predictions['False'] = 1 - predictions['True']

    return predictions
