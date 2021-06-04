
#import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import pickle
import json

def load_model(input_dir):
    generator = keras.Sequential(
    [
        keras.Input(shape=(128,)),
        # We want to generate 128 coefficients to reshape into a 7x7x128 map
        layers.Dense(7 * 7 * 128),
        layers.LeakyReLU(alpha=0.2),
        layers.Reshape((7, 7, 128)),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
    ],
    name="generator",
    )   
    generator.load_weights(os.path.join(input_dir, "gan_weights.h5"))
    return generator

def score_unstructured(model, data, query, **kwargs):
    print("Incoming content type params: ", kwargs)
    print("Incoming data type: ", type(data))
    print("Incoming query params: ", query)
    
    ## data is expected to be the number of images to generate
    random_latent_vectors = 128
   

    num_digits = json.loads(data)["num_digits"]
    ## need to parse data to int
  
    random_latent_vectors = tf.random.normal(shape=(num_digits, 128))
    rand_imgs = model(random_latent_vectors)
    rand_imgs *= 255
    rand_imgs.numpy()
    images = []
    # images["num_images": d]
    for i in range(num_digits):
            img = keras.preprocessing.image.array_to_img(rand_imgs[i])
            images.append(img)
        # img.save("generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch))
    return pickle.dumps(images)
   


