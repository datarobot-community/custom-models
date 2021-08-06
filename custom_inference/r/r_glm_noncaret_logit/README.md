## R Fit Template

The custom.R template includes a basic fit and init method that can be used to train a regression, binary or multilabel classification model.
The expected arguments to the fit method should remain the same, although the internal functionality can be tweaked to 
use different modeling or preprocessing techniques.

Inside you will find several commented out methods related to prediction behavior. Uncomment and implement provided methods to modify this behavior from the default.

Change the target value in the code below to Species to test multilabel classification.

### To run locally using 'drum'
Paths are relative to `datarobot-user-models` root:
`drum fit --code-dir model_templates/training/r_lang --input tests/testdata/iris.csv --target-type regression --target 'Petal.Width'`
If the command succeeds, your code is ready to be uploaded.

