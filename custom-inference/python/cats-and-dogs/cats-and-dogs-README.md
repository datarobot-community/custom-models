# Cats and Dogs Example

This folder includes an example of how to use DRUM with a Keras DNN leveraging GPUs for inference.  GPU is a costly resource for inference, but the point here to show how this can be accomplished via DRUM if the need arises.  The model was trained to classify an image as a cat or a dog.  This model originated in the model templates available within the [DRUM](https://github.com/datarobot/datarobot-user-models/tree/master/model_templates/inference/python3_keras_vizai_joblib)

Use google colab to follow alone with `Main_Script.ipynb`.

In this notebook you will 

* use DRUM to score data in batch
* use DRUM to serve the model as a rest enpoint leveraing GPUS for inference

Serving the model can be doen with either Flask, or Nginx and uwsgi.  Using Nginx, you will have to modify some files, but all of the content is highlighed in the notebook.    