# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 09:49:45 2021

@author: Administrator
"""
import pdb 
import time,datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import dateutil, pylab,random  
import numpy as np
from pylab import *  
from datetime import datetime,timedelta  
from pandas import Series,DataFrame
import pandas as pd
from math import radians, cos, sin, asin, sqrt  
import threading
import itertools
from multiprocessing import Pool
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
from keras.models import model_from_json

model_path='.\\model\\'
model_save_path=r'.\model_save\\'

def load_model(model_name):
    '''
    load model

    Parameters
    ----------
    model_name : model name.

    Returns
    -------
    new_model : model.

    '''
    model_file='%s%s.json'%(model_path,model_name)
    #weight path
    model_weight='%s%s.h5'%(model_path,model_name)
    
    with open(model_file, 'r') as file:
        model_json = file.read()
    # load model
    new_model = model_from_json(model_json)
    new_model.load_weights(model_weight)
    
    return new_model

def train_model(model_name,model,X_train,Y_train,X_test,Y_test):

    # early stoppping
    from keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=25, verbose=0,mode='auto')
    
    p1=time.time()
    
    model.fit(X_train, Y_train, epochs=800, batch_size=100, verbose=2,callbacks = [early_stopping],class_weight='auto', validation_data=(X_test, Y_test))
    p2=time.time()
    
    #Save training time
    run_time='run time:'+str(round((p2-p1),0))
    print(run_time)
    scores=model.evaluate(x=X_test,y=Y_test)
    scores_result='\n%s: %.2f%%'%(model.metrics_names[1],scores[1]*100)
    print(scores_result) 
    
   
    model_file=('%s%s.json'%(model_save_path,model_name))
    model_weight=('%s%s.h5'%(model_save_path,model_name))
    
    # save model
    model_json = model.to_json()
    with open(model_file, 'w') as file:
        file.write(model_json)
        
    # save weight
    model.save_weights(model_weight)
    return

def model_evaluation(model_name,wave_array):
    '''
    This function is used to evaluate the model

    Parameters
    ----------
    model_name : model name
    wave_array : evaluation data

    Returns
    -------
    model_acc : accuracy
    model_p : precision
    model_f1 : f1
    model_r : recall

    '''
    model_file=('%s%s.json'%(model_path,model_name))
    model_weight=('%s%s.h5'%(model_path,model_name))
    
    from keras.models import model_from_json
    with open(model_file, 'r') as file:
        model_json = file.read()
    inception_model = model_from_json(model_json)
    inception_model.load_weights(model_weight)
    
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    y_true=wave_array[:,18000].reshape(-1,1)
    X_test=wave_array[:,0:18000].reshape(wave_array[:,0:18000].shape[0],1,100,180)
    Y_testxx_inception = inception_model.predict(X_test)
    
    y_pred=np.argmax(Y_testxx_inception,axis=1).reshape(-1,1)
    
   
    model_acc=accuracy_score(y_true, y_pred)
    model_f1 = f1_score( y_true, y_pred, average='macro' )
    model_p = precision_score(y_true, y_pred, average='macro')
    model_r = recall_score(y_true, y_pred, average='macro')

    result='modelname:{} \naccuracy:{:.3%},precision:{:.3%},f1:{:.3%},recall:{:.3%}'.format(model_name,model_acc,model_p,model_f1, model_r)
    print(result)
    
    return model_acc,model_p,model_f1, model_r

