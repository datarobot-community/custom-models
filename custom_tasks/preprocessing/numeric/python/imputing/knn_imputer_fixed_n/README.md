# KNN (Nearest Neighbors) Missing Imputation
## For Numerics

### Overview
```
[x] accepts numeric inputs
[ ] accepts categorical inputs
[ ] outputs missing values
[x] works for binary targets (tested in DataRobot and confirmed)
[x] works for multiclass targets
[x] works for numeric targets
```
### Description

KNN Imputer was first supported by Scikit-Learn in December 2019 when it released its version 0.22. This imputer utilizes the k-Nearest Neighbors method to replace the missing values in the datasets with the mean value from the parameter ‘n_neighbors’ nearest neighbors found in the training set. By default, it uses a Euclidean distance metric to impute the missing values.

### References

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3668100/

http://www.stat.columbia.edu/~gelman/arm/missing.pdf

https://machinelearningmastery.com/knn-imputation-for-missing-values-in-machine-learning/

https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html

https://www.iriseekhout.com/missing-data/