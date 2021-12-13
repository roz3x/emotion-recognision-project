from keras.applications.vgg16 import VGG16
import numpy as np
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten

model = VGG16(include_top=False, input_shape=(64, 64, 3))
flat1 = Flatten()(model.layers[-1].output)
class1 = Dense(1024, activation='relu')(flat1)
output = Dense(5, activation='softmax')(class1)
model = Model(inputs=model.inputs, outputs=output)
model.summary()



from PIL import Image
import glob

dirs = ["fear", "Angry", "Happy", "Neutral", "disgust"]

res = {
        "fear" : [ 1, 0, 0, 0 ,0],
        "Angry": [ 0, 1, 0, 0, 0],
        "Neutral": [0, 0, 1, 0, 0],
        "Happy": [0, 0, 0, 1, 0],
        "disgust": [0, 0, 0, 0, 1]
    }

inputs = []
results = []

import cv2
import random

for ii in dirs:
    files = glob.glob("output/"+ii+"/*")
    random.shuffle(files)
    
    for f in files:
        img = cv2.imread(f)
        img = cv2.resize(img, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)

        img = np.array(img)
        inputs.append(img)
        results.append(res[ii])


inputs , results = np.array(inputs), np.array(results)
print(np.shape(inputs)[0], np.shape(results))


        






