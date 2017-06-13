import csv
import numpy as np
import pdb
import os
import csv
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Lambda, Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Cropping2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras.layers.pooling import MaxPooling2D
from PIL import Image, ImageOps
from keras import regularizers

def get_nvidia_model(num_outputs, l2reg):
    # Keras NVIDIA type model
    model =  Sequential()
    model.add(Lambda(lambda x : x / 255.0-0, input_shape=(480,640,3)))
    model.add(Cropping2D(cropping=((240,0),(0,0)))) # trim image to only see section with road

    model.add(Conv2D(24,(5,5), strides=(2,2), activation='relu', use_bias=True, kernel_initializer='glorot_normal', bias_initializer='zeros'))
    model.add(BatchNormalization())

    model.add(Conv2D(36,(5,5), strides=(2,2), activation='relu', use_bias=True, kernel_initializer='glorot_normal',bias_initializer='zeros'))
    model.add(BatchNormalization())

    model.add(Conv2D(48,(5,5), strides=(2,2), activation='relu', use_bias=True, kernel_initializer='glorot_normal',bias_initializer='zeros'))
    model.add(BatchNormalization())

    model.add(Conv2D(64,(3,3), activation='relu', use_bias=True, kernel_initializer='glorot_normal',bias_initializer='zeros'))
    model.add(BatchNormalization())

    model.add(Conv2D(64,(3,3), activation='relu', use_bias=True, kernel_initializer='glorot_normal',bias_initializer='zeros'))
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(1164, use_bias=True, kernel_initializer='glorot_normal',bias_initializer='zeros', kernel_regularizer=regularizers.l2(l2reg)))
    model.add(BatchNormalization())
    if num_outputs == 1:
        model.add(Dropout(0.2))


    model.add(Dense(100, use_bias=True, kernel_initializer='glorot_normal',bias_initializer='zeros', kernel_regularizer=regularizers.l2(l2reg)))
    model.add(BatchNormalization())
    if num_outputs == 1:
        model.add(Dropout(0.2))

    model.add(Dense(50, use_bias=True, kernel_initializer='glorot_normal',bias_initializer='zeros', kernel_regularizer=regularizers.l2(l2reg)))
    model.add(BatchNormalization())
    if num_outputs == 1:
        model.add(Dropout(0.2))

    model.add(Dense(10, use_bias=True, kernel_initializer='glorot_normal',bias_initializer='zeros',kernel_regularizer=regularizers.l2(l2reg)))
    model.add(BatchNormalization())
    if num_outputs == 1:
        model.add(Dropout(0.2))

    model.add(Dense(num_outputs, use_bias=True, kernel_initializer='glorot_normal',bias_initializer='zeros', kernel_regularizer=regularizers.l2(l2reg)))

    return model

def get_lstm_model(num_outputs, l2reg, num_inputs):
    # Keras NVIDIA type model
    model =  Sequential()

    model.add(Lambda(lambda x : x / 255.0, input_shape=(480,640, num_inputs)))
    model.add(Cropping2D(cropping=((240,0),(150,150)))) # trim image to only see section with road

    model.add(Conv2D(24,(5,5), strides=(2,2), activation='relu', use_bias=True, kernel_initializer='glorot_normal', bias_initializer='zeros'))
    model.add(BatchNormalization())

    model.add(Conv2D(36,(5,5), strides=(2,2), activation='relu', use_bias=True, kernel_initializer='glorot_normal',bias_initializer='zeros'))
    model.add(BatchNormalization())

    model.add(Conv2D(48,(5,5), strides=(2,2), activation='relu', use_bias=True, kernel_initializer='glorot_normal',bias_initializer='zeros'))
    model.add(BatchNormalization())

    model.add(Conv2D(64,(3,3), activation='relu', use_bias=True, kernel_initializer='glorot_normal',bias_initializer='zeros'))
    model.add(BatchNormalization())

    model.add(Conv2D(64,(3,3), activation='relu', use_bias=True, kernel_initializer='glorot_normal',bias_initializer='zeros'))
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(1164, use_bias=True, kernel_initializer='glorot_normal',bias_initializer='zeros', kernel_regularizer=regularizers.l2(l2reg)))
    model.add(BatchNormalization())
    if num_outputs == 1:
        model.add(Dropout(0.2))


    model.add(Dense(100, use_bias=True, kernel_initializer='glorot_normal',bias_initializer='zeros', kernel_regularizer=regularizers.l2(l2reg)))
    model.add(BatchNormalization())
    if num_outputs == 1:
        model.add(Dropout(0.2))

    model.add(Dense(50, use_bias=True, kernel_initializer='glorot_normal',bias_initializer='zeros', kernel_regularizer=regularizers.l2(l2reg)))
    model.add(BatchNormalization())
    if num_outputs == 1:
        model.add(Dropout(0.2))

    model.add(Dense(10, use_bias=True, kernel_initializer='glorot_normal',bias_initializer='zeros',kernel_regularizer=regularizers.l2(l2reg)))
    model.add(BatchNormalization())
    if num_outputs == 1:
        model.add(Dropout(0.2))

    model.add(Dense(num_outputs, use_bias=True, kernel_initializer='glorot_normal',bias_initializer='zeros', kernel_regularizer=regularizers.l2(l2reg)))

    return model
