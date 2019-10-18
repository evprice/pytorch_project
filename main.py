import numpy as np
import pandas as pd
import mxnet
import os
from xml.dom import minidom


import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


import keras
from os import listdir
from os.path import isfile, join


def get_cords():
    """Retrieves the coordinates (X,Y sets) of all objects being classified in
    the respective image, pairing it with its doc name. Returns a dictionary"""
    a = minidom.parse("Annotations\BloodImage_00000.xml")
    onlyfiles = [f for f in listdir("Annotations") if isfile(join("Annotations", f))]
    cords = {}

    for file in onlyfiles:
        a = minidom.parse(("Annotations/"+ file))
        zz = 0
        for obj in a.getElementsByTagName("object"):
            xminL = obj.getElementsByTagName("xmin")
            xmaxL = obj.getElementsByTagName("xmax")
            yminL = obj.getElementsByTagName("ymin")
            ymaxL = obj.getElementsByTagName("ymax")
            nameL = obj.getElementsByTagName("name")
            cords[file + str(zz)] = [xmaxL[0].firstChild.data,
            xminL[0].firstChild.data,
            ymaxL[0].firstChild.data,
            yminL[0].firstChild.data,
            nameL[0].firstChild.data]
            zz+=1
    return cords

image_set = get_cords()
img_cords = pd.DataFrame.from_dict(image_set)
img_cords = img_cords.transpose()
img_cords = img_cords.reset_index()
img_cords.columns = ["name", "xmin", "xmax", "ymin", "ymax", "type"]
img_cords.name = img_cords["name"].apply(lambda x: x[:16] + ".jpg")
img_cords
train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data
