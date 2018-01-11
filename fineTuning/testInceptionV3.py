
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import argparse
import imutils
import cv2

img_path = 'pumpkin.jpeg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)



# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model("inception.model")


preds = model.predict(x)
print (preds)
#print('Predicted:', decode_predictions(preds, top=3)[0])