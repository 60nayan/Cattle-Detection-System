
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.models import Sequential
#Defining parameters for the loader:
batch_size = 32
img_height = 180
img_width = 180

#Saving the model
# model_dl.save("model_dl.h5")

#Loading themodel
model_dl = keras.models.load_model("model_dl.h5") #look for local saved file
 
#Using a random cat picture found in the web
picture_url = "https://placekitten.com/g/200/300"
picture_path = tf.keras.utils.get_file("300", origin=picture_url)

img = keras.preprocessing.image.load_img(picture_path, target_size=(img_height, img_width))
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model_dl.predict(img_array)
score = tf.nn.sigmoid(predictions[0])

print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(10, 100 * np.max(score)))

