# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 23:40:27 2019

@author: tanma
"""
import load as l
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

model,_ = l.init()

def make_sentence(sentence):
    return [[sentence]]

def preprocessing(list_sentences,max_features = 20000, maxlen = 50):
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(list_sentences))
    list_tokenized_train = tokenizer.texts_to_sequences(list_sentences)
    
    X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
    return X_t

def prediction(x):
    x = preprocessing(make_sentence(x))
    list_classes = ["Toxic", "Severely Toxic", "Obscene", "Threat", "Insult", "Identity Hate"]      
    x = dict(zip(list_classes,100*model.predict([x,]).flatten()))
    return x

if __name__ == "__main__":
    x = "COCKSUCKER BEFORE YOU PISS AROUND ON MY WORK"
    print(prediction(x))