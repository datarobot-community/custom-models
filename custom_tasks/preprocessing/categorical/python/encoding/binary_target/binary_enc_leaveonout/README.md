# Leave-One-Out Target Encoding

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

This is very similar to target encoding but excludes the current row’s target when calculating the mean target for a level to reduce the effect of outliers.

### References

https://www.kaggle.com/c/caterpillar-tube-pricing/discussion/15748#143154