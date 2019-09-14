# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 02:26:05 2019

@author: tanma
"""

import tensorflow as tf
from keras.models import load_model   

def init():
    model = load_model('opt_model.h5')
    graph = tf.get_default_graph()
    
    return model,graph