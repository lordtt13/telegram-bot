# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 23:37:33 2019

@author: tanma
""" 
import tensorflow as tf
from keras.models import load_model   

def init():
    model = load_model('weights-improvement.hdf5')
    graph = tf.get_default_graph()
    
    return model,graph