import pandas as pd
from typing import List, Optional, Any, Dict

# """
# Custom hooks for prediction
# ---------------------------
# If drum's standard assumptions are incorrect for your model, **drum** supports several hooks
# for custom inference code.
# """
# def init(code_dir : Optional[str], **kwargs) -> None:
#     """

#     Parameters
#     ----------
#     code_dir : code folder passed in the `--code_dir` parameter
#     kwargs : future proofing
#     """


# def load_model(code_dir: str) -> Any:
#     """
#     Can be used to load supported models if your model has multiple artifacts, or for loading
#     models that **drum** does not natively support

#     Parameters
#     ----------
#     code_dir : is the directory where model artifact and additional code are provided, passed in

#     Returns
#     -------
#     If used, this hook must return a non-None value
#     """
#     return CustomModel(code_dir)


def transform(data: pd.DataFrame, model: Any) -> pd.DataFrame:
    """
    Intended to apply transformations to the prediction data before making predictions. This is
    most useful if **drum** supports the model's library, but your model requires additional data
    processing before it can make predictions

    Parameters
    ----------
    data : is the dataframe given to **drum** to make predictions on
    model : is the deserialized model loaded by **drum** or by `load_model`, if supplied

    Returns
    -------
    Transformed data
    """
    data = data = data.fillna(0)
    return data

# def score(data: pd.DataFrame, model: Any, **kwargs: Dict[str, Any]) -> pd.DataFrame:
#     """
#     This hook is only needed if you would like to use **drum** with a framework not natively
#     supported by the tool.

#     Parameters
#     ----------
#     data : is the dataframe to make predictions against. If `transform` is supplied,
#     `data` will be the transformed data.
#     model : is the deserialized model loaded by **drum** or by `load_model`, if supplied
#     kwargs : additional keyword arguments to the method
#     In case of classification model class labels will be provided as the following arguments:
#     - `positive_class_label` is the positive class label for a binary classification model
#     - `negative_class_label` is the negative class label for a binary classification model

#     Returns
#     -------
#     This method should return predictions as a dataframe with the following format:
#       Binary Classification: must have columns for each class label with floating- point class
#         probabilities as values. Each row should sum to 1.0
#       Regression: must have a single column called `Predictions` with numerical values

#     """
#     prediction = model.pipeline.predict(data)
#     return pd.DataFrame(prediction, columns=["Predictions"])


# def post_process(predictions: pd.DataFrame, model: Any) -> pd.DataFrame:
#     """
#     This method is only needed if your model's output does not match the above expectations

#     Parameters
#     ----------
#     predictions : is the dataframe of predictions produced by **drum** or by
#       the `score` hook, if supplied
#     model : is the deserialized model loaded by **drum** or by `load_model`, if supplied

#     Returns
#     -------
#     This method should return predictions as a dataframe with the following format:
#       Binary Classification: must have columns for each class label with floating- point class
#         probabilities as values. Each row
#     should sum to 1.0
#       Regression: must have a single column called `Predictions` with numerical values

#     """



