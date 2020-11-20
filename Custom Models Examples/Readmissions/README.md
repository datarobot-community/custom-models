# Custom Model Examples

This folder includes various examples of custom models using the well known  `Hospital Readmissions` dataset. There are three examples (Level 1 - 3) with each level presenting a more complicated example of custom model than the one previously to it.

## Important Links

For more information on how to use DRUM to test and deploy your custom models, follow the [link](https://github.com/datarobot-community/mlops-examples/tree/master/MLOps%20DRUM)

## Contents

#### Readmission_level_1

*High Level Overview:*

1. Fit a CatBoost classifier on top of the `Hospital Readmissions` dataset.
2. `custom.py` script needs to include Null value handling.
3. `custom.py` script needs to search for keyword `Diabetes|diabetes` in `diag_1_desc` column and create a new Boolean column.

#### Readmission_level_2

*High Level Overview:*

1. Preprocess Data using scikit-learn pipeline
2. Fit Keras model on the data
3. `custom.py` script needs to preprocess using the scikit-learn pipeline 
4. `custom.py` script needs to score using the Keras Model.

The extra difficulty here is that we need to define where DRUM is to find both of the preprocessing and the keras model in order for this custom model to work.

#### Readmission_level_3

The biostatisticians at ABC labs have a legacy model that they are using to predict probability to be readmitted into the hospital. They found that using an ensemble model between their own and the keras model, yields the best outcome. The result needs to be the average probability between the two models.

*High Level Overview:*

1. Preprocess Data using scikit-learn pipeline
2. Fit Keras model on the data
3. `custom.py` script needs to preprocess using the scikit-learn pipeline 
4. `custom.py` script needs to score using the Keras Model.
4. `custom.py` script needs to return the average probability as calculated from Keras + legacy model.