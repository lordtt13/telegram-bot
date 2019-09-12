# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 23:16:47 2019

@author: tanma
"""
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Embedding, SpatialDropout1D, Bidirectional
from keras.layers import Conv1D, GRU, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras.optimizers import Adam

def get_data(train, test, max_features = 20000, maxlen = 50):
    list_sentences_train = train["comment_text"].fillna("_na_").values
    list_sentences_test = test["comment_text"].fillna("_na_").values
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    y = train[list_classes].values

    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(list_sentences_train))
    list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
    list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
    
    X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
    X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)
    
    return X_t,X_te,y,tokenizer

def get_embedding_matrix(EMBEDDING_FILE, embed_size, tokenizer, max_features = 20000):
    embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE,'rb'))
    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    
    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    
    for word, i in word_index.items():
        if i >= max_features: 
            continue 
        embedding_vector = embeddings_index.get(word) 
    
    if embedding_vector is not None: 
        embedding_matrix[i] = embedding_vector 
        
    return embedding_matrix

def get_coefs(word,*arr): 
    return word, np.asarray(arr, dtype='float32')

def make_model(embed_size, embedding_matrix, max_features = 20000, maxlen = 50):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(128, return_sequences=True,dropout=0.1,recurrent_dropout=0.1))(x)
    x = Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool, max_pool])
    preds = Dense(6, activation="sigmoid")(x)
    
    model = Model(inp, preds)
    model.compile(loss='binary_crossentropy',optimizer=Adam(lr=1e-4),metrics=['accuracy'])
 
    return model

def train(model,X_t,y,MODEL_WEIGHTS_FILE):
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00000001)

    callbacks = [learning_rate_reduction,EarlyStopping('val_loss', patience=3), ModelCheckpoint(MODEL_WEIGHTS_FILE, save_best_only=True)]
    
    model.fit(X_t, y, batch_size=32, epochs=20, validation_split=0.1, callbacks=callbacks)
    
    model.save("toxic_model.h5")