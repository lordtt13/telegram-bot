# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 13:15:02 2019

@author: tanma
"""
import cv2
import cvlib as cv
import json
import numpy as np
import init as j
from keras.preprocessing.image import img_to_array

model,graph = j.init()

def integrator(image, model = model):
    image = cv2.imread(image)
    image = cv2.resize(image,(1024,1024))
    face, confidence = cv.detect_face(image)
    
    l = []
    for idx, f in enumerate(face):
       
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]
    
        face_crop = np.copy(image[startY:endY,startX:endX])
    
        face_crop = cv2.resize(face_crop, (128,128))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)
        global graph
        with graph.as_default():
            tags = model.predict(face_crop)
        if tags[0] > 0.5:
            case = {'Gender':'Male','Age':int(np.ceil(tags[1]))}
        else:
            case = {'Gender':'Female','Age':int(np.ceil(tags[1]))}

        
        l.append(case)
        
    return l

def beautify(l):
    string = ''
    for i in l:
        string += json.dumps(i)
        
    return string