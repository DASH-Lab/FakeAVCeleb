import pandas as pd
import numpy as np
import pickle
import re
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score



from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from keras.layers import Embedding, LSTM, Dropout, Dense, Input, Bidirectional, Flatten, Conv2D, MaxPooling2D, concatenate, Conv1D, MaxPooling1D
import keras.backend as K
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

import tensorflow as tf

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def precision_mmm(y_true, y_pred):
    #true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    #predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    #precision = (ff-true_positives) / ((ff-predicted_positives)+ K.epsilon() )
    return 1#int(tf.shape(y_true)[0])
    

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
   
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
    
    
def compile_model():
  # lstm_input = Input(shape=(MAX_SEQUENCE_LENGTH,))
  # x = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], mask_zero=True, input_length=MAX_SEQUENCE_LENGTH, trainable=False)(lstm_input)
  # x = Dropout(0.3)(x)
  # x = LSTM(64, return_sequences = True)(x)
  # x = Dropout(0.3)(x)
  # x = LSTM(64)(x)
  # x = Dropout(0.3)(x)
  # lstm_out = Dense(18, activation = 'relu')(x)

  cnn_input = Input(shape=(128, 128, 3))
  y = Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3))(cnn_input)
  # print(y.shape)
  y = MaxPooling2D(2, 2)(y)
  # print(y.shape)
  y = Conv2D(64, (3, 3), activation='relu')(y)
  # print(y.shape)
  y = MaxPooling2D(2, 2)(y)
  # print(y.shape)
  y = Conv2D(128, (3, 3), activation='relu')(y)
  # print(y.shape)
  y = MaxPooling2D(2, 2)(y)
  # print(y.shape)
  y = Conv2D(128, (3, 3), activation='relu')(y)
  # print(y.shape)
  y = MaxPooling2D(2, 2)(y)
  # print(y.shape)
  y = Flatten()(y)
  # print(y.shape)
  y = Dropout(0.3)(y)
  # print(y.shape)
  cnn_out = Dense(512, activation='relu')(y)

  # print(cnn_out.shape)
  # print("hashahashahs")

  cnn_input_spec = Input(shape=(128, 64, 3))
  y = Conv2D(32, (3, 3), activation='relu', input_shape=(128,64,3))(cnn_input_spec)
  # print(y.shape)
  y = MaxPooling2D(2, 2)(y)
  # print(y.shape)
  y = Conv2D(64, (3, 3), activation='relu')(y)
  # print(y.shape)
  y = MaxPooling2D(2, 2)(y)
  # print(y.shape)
  y = Conv2D(128, (3, 3), activation='relu')(y)
  # print(y.shape)
  y = MaxPooling2D(2, 2)(y)
  # print(y.shape)
  y = Conv2D(128, (3, 3), activation='relu')(y)
  # print(y.shape)
  y = MaxPooling2D(2, 2)(y)
  # print(y.shape)
  y = Flatten()(y)
  # print(y.shape)
  y = Dropout(0.3)(y)
  #print(y.shape)
  cnn_out_spec = Dense(512, activation='relu')(y)
  #print(cnn_out_spec.shape)

  concat_inp = concatenate([cnn_out, cnn_out_spec])
  z = Dense(256, activation='relu')(concat_inp)
  z = Dropout(0.3)(z)
  #print(z.shape)
  z = Dense(128, activation='relu')(z)
  z = Dropout(0.3)(z)
  z = Dense(64, activation='relu')(z)
  z = Dropout(0.3)(z)
  #print(z.shape)
  output = Dense(2, activation='sigmoid')(z)

  model = Model(inputs=[cnn_input, cnn_input_spec], outputs=[output])
  adam = Adam(lr=0.001, decay=1e-5)
  
  
  
  model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['acc',f1_m,tf.keras.metrics.Precision(),tf.keras.metrics.Recall(), recall_m])
  return model
  
  
word2vec_model = compile_model()
word2vec_model.summary()
es = EarlyStopping(patience=5)
csv_logger = CSVLogger('model_history_log_model_lstm.csv', append=True)
check_point = ModelCheckpoint(filepath='best_lstm_model_todate_new', save_best_only=True, save_weights_only=True)

from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rescale=1.0/255,)
train_fram = datagen.flow_from_directory('data/FRAMS',target_size=(128,128),class_mode='binary',batch_size=16)
test_fram = datagen.flow_from_directory('testdata/FRAMS',target_size=(128,128),class_mode='binary',batch_size=16)
train_spec = datagen.flow_from_directory('data/SPEC',target_size=(128,64),class_mode='binary',batch_size=16)
test_spec = datagen.flow_from_directory('testdata/SPEC',target_size=(128,64),class_mode='binary',batch_size=16)


from keras.utils import to_categorical

def combine_gen(gens1,gens2):
    while True:
        lab = []
        g1 = next(gens1)
        g2 = next(gens2)
        #print(g1[0])
        g1s = []
        g2s = []
        for xx in range(len(g1[1])):
            
            if g1[1][xx] == 0 and g2[1][xx] == 0:
                lab.append(0)
                g1s.append(g1[0][xx])
                g2s.append(g2[0][xx])
        ss = len(g1s)
        ff = 0
        for xx in range(len(g1[1])):
            if len(g1s)<(2*ss):
                if g1[1][xx] == 0 and g2[1][xx] == 0:
                    ff+=1
                else:
                    lab.append(1)
                    g1s.append(g1[0][xx])
                    g2s.append(g2[0][xx])
        print(len(lab))
        #g2s = g2s[:len(g1s)]
        #print(np.asarray(g1s).shape)
        if len(g1s)>0 and len(g1s)==(2*ss):
            yield [np.asarray(g1s),np.asarray(g2s)],to_categorical(lab)#,g2[1]
        
word2vec_model.load_weights("best_lstm_model_todate_new")
word2vec_multi_modal_model = word2vec_model.fit_generator(combine_gen(train_fram, train_spec), epochs=100, steps_per_epoch=115680/16, validation_steps=4313/16,
                            validation_data=combine_gen(test_fram, test_spec), callbacks = [ csv_logger, check_point])

