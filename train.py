# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 14:33:13 2020
    This code is used for model training.
    Before executing the program, please make sure that the data is complete 
    and placed in the data subdirectory.
    This page code is the supporting code of paper 
real time recognition classification of near earth events 
based on machine learning.
    In order to facilitate expert review, the code removed 
parallel GPU support and some unnecessary experimental debugging
code.
    Required package version, please refer to Required.txt
    Data format description,please refer to Data.txt
    
@author: Luozhao Jia  
email: lezhao.jia@gmail.com  123@eqha.gov.cn

"""


import numpy as np
from keras import backend
from keras.utils.np_utils import to_categorical
import h5py
import os,sys
import model_definition
from model import train_model
backend.set_image_data_format('channels_first')

#Data file path
train_file=r'.\data\train_set.h5'
validation_file=r'.\data\validation_set.h5'

if __name__ == '__main__':
    if(~os.path.exists(train_file)):
        print('The data file does not exist, please download and put it in the data directory.\n')
        print('If there is no data, please run independent_test.py !')

        sys.exit()
    
    train_set=np.empty(0)
    validation_set=np.empty(0)
    
    #Read data into memory
    with h5py.File(train_file, 'r') as hf:
        train_set = hf['earthquake'][:]
    
    with h5py.File(validation_file, 'r') as hf:
        validation_set = hf['earthquake'][:]
        
    # Set random seeds to ensure consistency of each learning
    seed = 7
    np.random.seed(seed)
     
    #data preparation
    
    np.random.shuffle(train_set)
    np.random.shuffle(validation_set)
    
    #divide the data into data sets and flag bits
    X_train=train_set[:,0:18000].reshape(train_set.shape[0],1,100,180)
    Y_train=train_set[:,18000].reshape(-1,1)
    
    X_test=validation_set[:,0:18000].reshape(validation_set.shape[0],1,100,180)
    Y_test=validation_set[:,18000].reshape(-1,1)
    
    
    Y_train=to_categorical(Y_train, num_classes=3)
    Y_test=to_categorical(Y_test, num_classes=3)
    
    
    #Only one model can be trained at the same time. 
    #Please comment the other two
    
    #model 1 ,Based on VGG network 92.19%
    model_name='vgg'
    model= model_definition.create_model_lcnn2()
    train_model(model_name,model,X_train,Y_train,X_test,Y_test)
    # The model is saved in the model_save directory
    
    #model 2  ,Based on RES network 91.2%
    model_name='resmodel'
    model= model_definition.creatcnn501()
    train_model(model_name,model,X_train,Y_train,X_test,Y_test)
    # The model is saved in the model_save directory
    
    #model 3 ,Based on Inception network 90.60%
    model_name='inception6'
    model= model_definition.create_model_inception6()
    train_model(model_name,model,X_train,Y_train,X_test,Y_test)
    # The model is saved in the model_save directory



