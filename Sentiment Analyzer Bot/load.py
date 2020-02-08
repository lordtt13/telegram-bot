# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 23:37:33 2019

@author: tanma
""" 
import tensorflow as tf
from tensorflow.keras.models import load_model   

def init():
    model = load_model('toxic_model.h5')
    graph = tf.compat.v1.get_default_graph()
    
    return model,graph