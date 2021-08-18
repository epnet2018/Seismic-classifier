# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 10:50:43 2021

@author: Administrator
"""

import pdb 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import dateutil, pylab,random  
import numpy as np
from pylab import *  
from datetime import datetime,timedelta  
from pandas import Series,DataFrame
import pandas as pd
from math import radians, cos, sin, asin, sqrt  
import itertools
import numba as nb
import tensorflow as tf
import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, Activation
from keras.layers import Flatten
from keras.layers.convolutional import  Conv2D
from keras.layers.convolutional import  Conv1D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import GlobalAveragePooling1D
from keras.utils import np_utils
from keras import backend
from keras.utils.np_utils import to_categorical
from sklearn import metrics
from keras import backend
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.initializers import RandomNormal
from keras import optimizers
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.utils import multi_gpu_model
from keras.layers import add,Input,Dense,Activation
from keras.models import Model
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from keras.layers import add, Flatten, Activation
from keras.optimizers import SGD,Adam
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.callbacks import TensorBoard
from keras.utils import plot_model
from tensorflow.python.client import device_lib
import h5py
from sklearn.preprocessing import normalize
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy import stats

backend.set_image_data_format('channels_first')

def ResBlock(x,hidden_size1,hidden_size2):
    '''
    Residual block layer
    '''
    r=Dense(hidden_size1,activation='relu')(x)  
    r=Dense(hidden_size2)(r)  
    if x.shape[1]==hidden_size2:
        shortcut=x
    else:
        shortcut=Dense(hidden_size2)(x)  # shortcut connections
    o=add([r,shortcut])
    o=Activation('relu')(o) 
    return o

def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    '''
    Test batchNormalization
    '''
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
 
    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, name=conv_name,kernel_regularizer=keras.regularizers.l2(0.0001),activation='relu')(x)
    x = Dropout(0.1)(x)
    # Try using batchnormalization
    # x = BatchNormalization(axis=3, name=bn_name)(x)
    # x = Activation('relu')(x)

    return x
  
def Conv_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    '''
    Conversion block layer
    '''
    x = Conv2d_BN(inpt, nb_filter=nb_filter[0], kernel_size=(1, 1), strides=strides, padding='same')
    #x = Conv2d_BN(x, nb_filter=nb_filter[1], kernel_size=(3, 3), padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter[2], kernel_size=(1, 1), padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=nb_filter[2], strides=strides, kernel_size=kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x

def creatcnn501():
    '''
    Residual structure based network optimization
    Total params: 18,183
    accuracy: 91.63%
    run time:1577.0
    0.916347945474 0.920907678089 0.919291239136 0.91815077279 
    After using dropout , accuracy: 91.69%, run time:5002.0 s 
    0.91693635383 0.922487501896 0.915560760688 0.910166767122  
    '''

    inpt = Input(shape=(1,100,180))
    x = ZeroPadding2D((3, 3))(inpt)
    x = Conv2d_BN(x, nb_filter=16, kernel_size=(5, 5), strides=(2, 2), padding='valid')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
 
    x = Conv_Block(x, nb_filter=[8, 8, 24], kernel_size=(5, 5), strides=(1, 1), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=[8, 8, 24], kernel_size=(3, 3))

    x = Conv_Block(x, nb_filter=[8, 8, 16], kernel_size=(2, 2), strides=(2, 2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=[8, 8, 16], kernel_size=(2, 2))
    
    x = Conv_Block(x, nb_filter=[4, 4, 16], kernel_size=(2, 2), strides=(2, 2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=[4, 4, 16], kernel_size=(2, 2))

    x = Conv_Block(x, nb_filter=[2, 2, 16], kernel_size=(2, 2), strides=(2, 2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=[2, 2, 16], kernel_size=(2, 2))
       
    x = AveragePooling2D(pool_size=(3, 3),padding='same')(x)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(3, activation='softmax')(x)
    model = Model(inputs=inpt, outputs=x)
    #sgd = SGD(decay=0.0001, momentum=0.9)
    #model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model 

def create_model_inception6():
    '''
    Inception structure based network optimization
    run time:12998.0
    accuracy: 90.75%
    '''

    from keras.layers import Conv2D, MaxPooling2D, Input
    inputs = Input(shape=(1,100,180))
  
    tower_1 = Conv2D(32, (5, 5), padding='same', activation='relu')(inputs)
    tower_1 = MaxPooling2D(pool_size=(2, 2))(tower_1)
    tower_1 = Conv2D(32, (3, 3), padding='same', activation='relu')(tower_1)
    tower_1 = Dropout(0.2)(tower_1)
    
    tower_2 = Conv2D(32, (5, 5), padding='same', activation='relu')(inputs)
    tower_2 = MaxPooling2D(pool_size=(2, 2))(tower_2)
    tower_2 = Conv2D(32, (3, 3), padding='same', activation='relu')(tower_2)
    tower_2 = Dropout(0.2)(tower_2)
    
    tower_3 = Conv2D(32, (5, 5), padding='same', activation='relu')(inputs)
    tower_3 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(tower_3)
    tower_3 = Conv2D(32, (3, 3), padding='same', activation='relu')(tower_3)
    tower_3 = Dropout(0.2)(tower_3)
    
    from keras import layers
    x = layers.concatenate([tower_1, tower_2, tower_3], axis=2)
    x = GlobalAveragePooling2D()(x)
    
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(3, activation='softmax')(x)
    
    outputs = Dense(3, activation='softmax', name='predictions')(x)
    
    model = Model(inputs, outputs, name='inception_v3')
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    
    return model


def create_model_lcnn2():
    '''
    Vggnet structure based network optimization
    Run time:2287.0
    accuracy: 92.21% 
    '''
   
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(1,100,180), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(6, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(6, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(units=36, activation='relu'))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=3, activation='softmax'))
    

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
   
    return model

