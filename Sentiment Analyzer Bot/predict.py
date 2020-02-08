# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 23:40:27 2019

@author: tanma
"""
import load as l
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

model,_ = l.init()

def make_sentence(sentence):
    return [sentence]

def load_lines(filename):
    TRAIN_DATA_FILE='E:\\Misc\\Toxic Model\\train.csv'

    train = pd.read_csv(TRAIN_DATA_FILE)
    list_sentences_train = train["comment_text"].fillna("_na_").values
    return list(list_sentences_train)

def preprocessing(list_sentences,sentence,max_features = 20000, maxlen = 50):
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(list_sentences))
    list_tokenized_train = tokenizer.texts_to_sequences(sentence)
    
    X_t = pad_sequences(list_tokenized_train, maxlen = maxlen)
    return X_t

def make_training_data(x, filename):
    data = load_lines(filename)
    return preprocessing(data,make_sentence(x))

def prediction(x):
    x = make_training_data(x, 'mal.txt')
    list_classes = ["Toxic", "Severely Toxic", "Obscene", "Threat", "Insult", "Identity Hate"]     
    x = dict(zip(list_classes, 100*model.predict([x,]).flatten()))
    return x

if __name__ == "__main__":
    x = "COCKSUCKER BEFORE YOU PISS AROUND ON MY WORK"
    print(x, prediction(x))