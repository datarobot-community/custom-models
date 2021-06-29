## Scala Custom Inference

Ths custom inference model was written in Scala, using XGBoost4j to predict Iris Species (Binary Version).  The main class, `XGBoostPredictor` inherits the `BasePredictor` class from DRUM.  You can see the code [here](src/main/scala/XGBoostPredictor.scala).  Training Code was included as well.  

The serialized version of the model is already available, but if you would like to train and save it to `custom-model/xgb-model` run the following from commend line.  

`java -jar custom-model/custom-scala-assembly-0.1.0.jar ./custom-model/xgb-model`

To run this model with `DRUM` export the following environment variables.  


`export DRUM_JAVA_CUSTOM_CLASS_PATH=/full/path/to/custom-model/custom-scala-assembly-0.1.0.jar`

`export DRUM_JAVA_CUSTOM_PREDICTOR_CLASS=custom.XGBoostPredictor`

Now run with DRUM 

`drum score --code-dir ./custom-model --input data/iris_binary_training.csv --target-type binary --positive-class-label 1 --negative-class-label 0`

## requirements 

* java 11

To build the jar youl will need sbt