# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 02:32:19 2019

@author: tanma
"""

import tensorflow as tf
from keras.models import load_model   

def init():
    model = load_model('model.h5')
    graph = tf.get_default_graph()
    
    return model,graph