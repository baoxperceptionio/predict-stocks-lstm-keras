# -*- coding: utf-8 -*-
# Brief: A Seq2Seq Model Implementation by keras Recurrent Neural Network LSTM.
# Author: Tateo_YANAGI @soarcloud.com
#
import numpy as np
import glog
import os
import tensorflow as tf

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint

import msgpack_numpy as msg_np

class LstmModel:
  def __init__(self, epochs=30000, length_of_sequences=1, bert_dim=768):
    self.length_of_sequences = length_of_sequences
    self.bert_dim = bert_dim
    self.batch_size = 1024
    self.train_sample_num = 0
    self.epochs = epochs
    self.output_dir = 'checkpoint'
    self.model = None
 
  def get_callbacks(self):
    """Return callbacks to pass into the Model.fit method
    Note: This simply returns statically instantiated callbacks. In the
    future it could be altered to allow for callbacks that are specified
    and configured via a training config.
    """

    fpath_weights = os.path.join(self.output_dir, str(self.length_of_sequences) + '_{epoch:05d}.h5')
    # save_freq is computed on batches. not epcho
    model_checkpoint = ModelCheckpoint(
        filepath=fpath_weights, verbose=True, save_freq=5000
    )
    callbacks = [model_checkpoint]

    return callbacks

  # prepare data
  def load_training_data(self, htb_msgpack_file, bth_msgpack_file):
    with open(htb_msgpack_file, 'rb') as f:
      htb_vecs = msg_np.unpackb(f.read())
    with open(bth_msgpack_file, 'rb') as f:
      btb_vecs = msg_np.unpackb(f.read())
    self.length_of_sequences = np.shape(htb_vecs)[1]
    self.train_sample_num = np.shape(htb_vecs)[0]
    self.bert_dim = np.shape(htb_vecs)[2]
    glog.info('self.length_of_sequences=' + str(self.length_of_sequences))
    glog.info('self.train_sample_num=' + str(self.train_sample_num))
    glog.info('self.bert_dim=' + str(self.bert_dim))
    glog.check_eq(self.length_of_sequences, np.shape(btb_vecs)[1])
    glog.check_eq(self.train_sample_num, np.shape(btb_vecs)[0])
    glog.check_eq(self.bert_dim, np.shape(btb_vecs)[2])
    return htb_vecs, btb_vecs

  # stress test
  def random_training_data(self):
    self.length_of_sequences = 5
    self.bert_dim = 300
    self.train_sample_num = 100
    X = np.random.rand(self.train_sample_num, self.length_of_sequences, self.bert_dim)
    Y = np.random.rand(self.train_sample_num, 1, self.bert_dim)
    return X, Y
 
  # make model
  def create_model(self):
    if self.length_of_sequences == 0:
        glog.error("self.length_of_sequences = 0. you need to load data first (or use random_data())")
        return None
    self.model = Sequential()
    # https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/
    # Expected input batch shape: (batch_size, timesteps, data_dim).
    input_shape=(self.length_of_sequences, self.bert_dim)
    batch_input_shape=(None, self.length_of_sequences, self.bert_dim)
    # units: Positive integer, dimensionality of the output space 
    #Model.add(LSTM(units=self.bert_dim, input_shape=input_shape, return_sequences=False))
    self.model.add(LSTM(units=self.bert_dim, input_length=self.length_of_sequences, input_dim=self.bert_dim, return_sequences=False))
    self.model.add(Dense(units=self.bert_dim, activation='relu'))
    self.model.add(Dense(units=self.bert_dim))
    self.model.add(Activation("linear"))
    #Model.compile(loss=tf.keras.losses.CosineSimilarity(axis=0), optimizer="sgd")
    self.model.compile(loss="mean_squared_error", optimizer="sgd")
    self.model.summary()
    return True
  def load_model_from_file(self, file):
    self.create_model()
    self.model.load_weights(file)
  # do learning
  def train(self, x_train, y_train, x_val, y_val) :
    tf.debugging.set_log_device_placement(True)
    self.model.fit(x_train, y_train, batch_size=self.batch_size, validation_data=(x_val, y_val), epochs=self.epochs, workers=16, callbacks=self.get_callbacks())
    return True
  def predict(self, x_test):
      return self.model.predict(x_test)