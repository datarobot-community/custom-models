## Python Keras Image Object Detection Custom Inference Model Template

This model is intended to work with the [Python 3 Keras Drop-In Environment](../../public_dropin_environments/python3_keras/).

## Instructions
Create a new custom model with `Unstructured` Target Type, add the files in the model folder and use the [Python 3 Keras Drop-In Environment] with it

Test with custom-models-wip/drum_overview/data/image_b64.txt

### To run locally using 'drum'
Paths are relative to `./custom-models`:  
`drum score --code-dir ./custom_inference/python/image_object_detection/model --target-type unstructured --input ./drum_overview/data/image_b64.txt --verbose`
