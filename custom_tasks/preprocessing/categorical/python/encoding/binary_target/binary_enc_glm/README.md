# GLM Target Encoding

### Overview
```
[ ] accepts numeric inputs
[x] accepts categorical inputs
[ ] outputs missing values
[x] works for binary targets (tested in DataRobot and confirmed)
[ ] works for multiclass targets
[ ] works for numeric targets
```
### Description

**This implementation is hard-coded to handle binary targets ONLY!**

This is a supervised encoder similar to TargetEncoder or MEstimateEncoder, but there are some advantages: 

1) Solid statistical theory behind the technique. Mixed effects models are a mature branch of statistics. 
2) No hyper-parameters to tune. The amount of shrinkage is automatically determined through the estimation process. In short, the less observations a category has and/or the more the outcome varies for a category then the higher the regularization towards “the prior” or “grand mean”. 
3) The technique is applicable for both continuous and binomial targets. If the target is continuous, the encoder returns regularized difference of the observation’s category from the global mean. If the target is binomial, the encoder returns regularized log odds per category.

In comparison to JamesSteinEstimator, this encoder utilizes generalized linear mixed models from statsmodels library.

### References

https://faculty.psau.edu.sa/filedownload/doc-12-pdf-a1997d0d31f84d13c1cdc44ac39a8f2c-original.pdf