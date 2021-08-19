# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 14:33:13 2020
    This code is used for the evaluation test of the model, 
    and 4 evaluation indicators of the model are given.
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
    
@author: luozhao jia  
email: lezhao.jia@gmail.com  123@eqha.gov.cn

"""

import numpy as np
from keras import backend
import h5py
from model import model_evaluation
backend.set_image_data_format('channels_first')

#Independent test set file path,The default is the example data, please download the official data to the data directory
#train_file='.\\data\\independent_test_set.h5'
#train_file='.\\data\\validation_set.h5'
#train_file='.\\data\\train_set.h5'
train_file='.\\data\\example_data.h5'
backend.set_image_data_format('channels_first')



if __name__ == '__main__':
    wave_array=np.empty(0)

    #Read data into memory
    with h5py.File(train_file, 'r') as hf:
        wave_array = hf['earthquake'][:]
    
    # Set random seeds to ensure consistency of each learning
    seed = 7
    np.random.seed(seed)
    np.random.shuffle(wave_array)
    
    #Calculate metrics globally by counting the total true positives, false negatives and false positives.
    model_name='vgg'
    model_evaluation(model_name,wave_array)
    
    model_name='inception6'
    model_evaluation(model_name,wave_array)
    
    model_name='resmodel'
    model_evaluation(model_name,wave_array)

