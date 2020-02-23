# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 23:40:27 2019

@author: tanma
"""
import pickle
import load as l

from tensorflow.keras.preprocessing.sequence import pad_sequences


tokenizer = pickle.load(open("tokenizer_instance.pickle","rb"))

model, _ = l.init()

def make_sentence(sentence):
    return [sentence]


def preprocessing(sentence, max_features = 20000, maxlen = 50, tokenizer = tokenizer):
    list_tokenized_train = tokenizer.texts_to_sequences(sentence)
    
    X_t = pad_sequences(list_tokenized_train, maxlen = maxlen)
    return X_t

def prediction(x):
    x = preprocessing(make_sentence(x))
    list_classes = ["Toxic", "Severely Toxic", "Obscene", "Threat", "Insult", "Identity Hate"]     
    x = str(dict(zip(list_classes, 100*model.predict([x,]).flatten())))
    return x