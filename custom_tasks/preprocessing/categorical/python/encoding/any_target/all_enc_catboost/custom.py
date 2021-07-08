import category_encoders as ce
import pandas as pd
from pathlib import Path
import pickle


def fit(X, y, output_dir, **kwargs):
    """ This hook defines how DataRobot will train this task. Even transform tasks need to be trained to learn/store information from training data
    DataRobot runs this hook when the task is being trained inside a blueprint.
    As an output, this hook is expected to create an artifact containg a trained object [in this example - a pre-fit CatBoost encoder], that is then used to transform new data.
    The input parameters are passed by DataRobot based on project and blueprint configuration.

    Parameters
    -------
    X: pd.DataFrame
        Training data that DataRobot passes when this task is being trained.
    y: pd.Series
        Project's target column (None is passed for unsupervised projects).
    output_dir: str
        A path to the output folder; the artifact [a pickle file containing a pre-fit CatBoost Encoder] must be saved into this folder to be re-used in transform().

    Returns
    -------
    None
        fit() doesn't return anything, but must output an artifact (typically containing a trained object) into output_dir
        so that the trained object can be used during scoring inside transform()
    """

    # Transform categorical columns into a numeric transformation using Weight of Evidence
    encoder_catboost = ce.CatBoostEncoder(cols=X.columns)
    encoder_catboost.fit(X,y)

    # dump the trained object 
    # into an artifact [in this example - woe.pkl]
    # and save it into output_dir so that it can be used later to impute on new data
    output_dir_path = Path(output_dir)
    if output_dir_path.exists() and output_dir_path.is_dir():
        with open("{}/catboost.pkl".format(output_dir), "wb") as fp:
            pickle.dump(encoder_catboost, fp)


def transform(data, transformer): 
    """ This hook defines how DataRobot will use the trained object from fit() to transform new data.
    DataRobot runs this hook when the task is used for scoring inside a blueprint. 
    As an output, this hook is expected to return the transformed data.
    The input parameters are passed by DataRobot based on dataset and blueprint configuration.

    Parameters
    -------
    data: pd.DataFrame
        Data that DataRobot passes for transformation.
    transformer: Any
        Trained object, extracted by DataRobot from the artifact created inside fit().
        In this example, it's a pickle file containing a pre-fit CatBoost Encoder.
    
    Returns
    -------
    pd.DataFrame
        Returns a dataframe with transformed data.
    """

    return transformer.transform(data).fillna(0)