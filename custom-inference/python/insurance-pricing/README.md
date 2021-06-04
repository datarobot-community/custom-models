# Insurance Pricing Example

This folder includes an example of a custom model using DRUM to predict loss on insurance claims. For simplicity, you can use google colab to follow alone with `Main_Script.ipynb`.

The point of this example is to highlight DRUM's unstructured mode.  DRUM relaxing validation of the input and output, allowing for flexibility in terms of the model that can be hosted and morever, the way data is sent to and returned from DRUM.  

In unstructured mode, this model will return shap values and loss prediction all by leverage avro for serializsation of data.  

Avro has a JSON like data model, but can be represented as either JSON or in a compact binary form.  
* It comes with a very sophisticated schema description language that describes data. 
* It has a direct mapping to and from JSON. 
* It has a very compact format. 
* The bulk of JSON, repeating every field name with every single record, is what makes JSON inefficient for high-volume usage

