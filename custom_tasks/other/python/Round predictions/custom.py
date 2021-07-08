from typing import List, Optional
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline


def fit(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: str,
    class_order: Optional[List[str]] = None,
    row_weights: Optional[np.ndarray] = None,
    **kwargs,
) -> None:

    estimator = pipeline(X)
    estimator.fit(X, y)

    output_dir_path = Path(output_dir)
    if output_dir_path.exists() and output_dir_path.is_dir():
        with open("{}/artifact.pkl".format(output_dir), "wb") as fp:
            pickle.dump(estimator, fp)


class RoundInput():
    """
    Goal is to round the output of a prior model, so using those unrounded
    predictions as inputs here.
    """

    def __init__(self, X):
        self.X = X
        
    def fit(self, X, y=None, **kwargs):
        self.X = round(X)
        return self

    def transform(self, X):
        return np.array(round(X[X.columns[0]])).reshape(-1, 1)


class EmptyEstimator():
    """
    This is empty because the rounding is done in the above step of the pipeline.
    Still need this for the pipeline to run though.
    """

    def fit(self, X, y):
        return self

    def predict(self, data: pd.DataFrame):
        return data[:,0]


def pipeline(X):
    return Pipeline(steps=[("preprocessing", RoundInput(X)), ("model", EmptyEstimator())])


def score(data: pd.DataFrame, model, **kwargs) -> pd.DataFrame:
    return pd.DataFrame(data=model.predict(data), columns = ['Predictions'])
