# -*- coding: utf-8 -*-
# Brief: A Seq2Seq Model Implementation by keras Recurrent Neural Network LSTM.
# Author: Tateo_YANAGI @soarcloud.com
#
import numpy as np
import glog
import os
import tensorflow as tf

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint

import msgpack_numpy as msg_np

class LstmModel:
  def __init__(self, epochs=30000, length_of_sequences=1, bert_dim=768, output_dir='checkpoint'):
    self.length_of_sequences = length_of_sequences
    self.bert_dim = bert_dim
    self.batch_size = 1024
    self.train_sample_num = 0
    self.epochs = epochs
    self.output_dir = output_dir
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
    early_stop = tf.keras.callbacks.EarlyStopping(
      monitor='loss', min_delta=1e-6, patience=500, verbose=0,
      mode='auto', baseline=None, restore_best_weights=False)

    callbacks = [model_checkpoint]

    return callbacks

  # prepare data
  def load_training_data(self, htb_msgpack_file, bth_msgpack_file):
    with open(htb_msgpack_file, 'rb') as f:
      htb_vecs = msg_np.unpackb(f.read())
      for it in range(htb_vecs.shape[0]):
        norm = np.linalg.norm(htb_vecs[it, :])
        if norm > 1 + 1e-4 or norm < 1 - 1e-4:
          glog.error('werid htb_vecs norm=' + str(norm))
    with open(bth_msgpack_file, 'rb') as f:
      bth_vecs = msg_np.unpackb(f.read())
      for it in range(bth_vecs.shape[0]):
        norm = np.linalg.norm(bth_vecs[it, :])
        if norm > 1 + 1e-4 or norm < 1 - 1e-4:
          glog.error('werid btb_vecs norm=' + str(norm))

    self.length_of_sequences = np.shape(htb_vecs)[1]
    self.train_sample_num = np.shape(htb_vecs)[0]
    self.bert_dim = np.shape(htb_vecs)[2]
    glog.info('self.length_of_sequences=' + str(self.length_of_sequences))
    glog.info('self.train_sample_num=' + str(self.train_sample_num))
    glog.info('self.bert_dim=' + str(self.bert_dim))
    glog.check_eq(self.length_of_sequences, np.shape(bth_vecs)[1])
    glog.check_eq(self.train_sample_num, np.shape(bth_vecs)[0])
    glog.check_eq(self.bert_dim, np.shape(bth_vecs)[2])
    # split data
    # shuffle
    randomList = np.arange(htb_vecs.shape[0])
    np.random.shuffle(randomList)
    htb_vecs = htb_vecs[randomList, :, :]
    bth_vecs = bth_vecs[randomList, :, :]
    val_percetage = 0.05
    x_train = htb_vecs[int(htb_vecs.shape[0]*val_percetage):, :]
    y_train = bth_vecs[int(bth_vecs.shape[0]*val_percetage):, :]
    y_train = y_train[:, -1, :]
    y_train = y_train[:, np.newaxis, :]
    if y_train.shape[1] != 1:
      glog.error('weird y_train shape ' + str(y_train.shape))
    for it in range(y_train.shape[0]):
      norm = np.linalg.norm(y_train[it, 0, :])
      if norm > 1 + 1e-4 or norm < 1 - 1e-4:
        glog.error('werid btb_vecs norm=' + str(norm))

    x_val = htb_vecs[:int(htb_vecs.shape[0]*val_percetage), :]
    y_val = bth_vecs[:int(bth_vecs.shape[0]*val_percetage), :]
    y_val = y_val[:, -1, :]
    y_val = y_val[:, np.newaxis, :]
    if y_val.shape[1] != 1:
      glog.error('y_val shape ' + str(y_val.shape))
    for it in range(y_val.shape[0]):
      norm = np.linalg.norm(y_val[it, 0, :])
      if norm > 1 + 1e-4 or norm < 1 - 1e-4:
        glog.error('werid btb_vecs norm=' + str(norm))
    return x_train, y_train, x_val, y_val

  # stress test
  def random_training_data(self):
    self.length_of_sequences = 5
    self.train_sample_num = 10000
    x_train = np.random.rand(self.train_sample_num, self.length_of_sequences, self.bert_dim)
    y_train = x_train[:, -1, :] # so we can get a perfect fit
    y_train = y_train[:, np.newaxis, :]
    x_val = np.random.rand(int(self.train_sample_num * 0.1), self.length_of_sequences, self.bert_dim)
    y_val = x_val[:, -1, :] # so we can get a perfect fit
    y_val = y_val[:, np.newaxis, :]

    return x_train, y_train, x_val, y_val
 
  # make model
  def create_model(self):
    if self.length_of_sequences == 0:
        glog.error("self.length_of_sequences = 0. you need to load data first (or use random_data())")
        return None
    self.model = Sequential()
    # https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/
    # Expected input batch shape: (batch_size, timesteps, data_dim).
    input_shape=(self.length_of_sequences, self.bert_dim)
    # units: Positive integer, dimensionality of the output space 
    #Model.add(LSTM(units=self.bert_dim, input_shape=input_shape, return_sequences=False))
    lstm_out_units = self.bert_dim # this does not neccessarily equal to bert dim
    # see the # of param calculation in https://datascience.stackexchange.com/questions/10615/number-of-parameters-in-an-lstm-model
    self.model.add(LSTM(units=lstm_out_units, input_length=self.length_of_sequences, input_dim=self.bert_dim, return_sequences=False))
    self.model.add(Dropout(0.2))
    self.model.add(Dense(units=self.bert_dim, activation='relu'))
    # self.model.add(Activation("linear"))
    # cosine_similarity mean_squared_error
    self.model.compile(loss="cosine_similarity", optimizer="sgd")

    self.model.summary()
    return True
  def load_model_from_file(self, file):
    self.create_model()
    self.model.load_weights(file)
  def load_last_checkpoint(self, checkpoint_folder):
    files = os.listdir(checkpoint_folder)
    files.sort()
    files.reverse()
    last_model_file = None
    for file in files:
        if os.path.isfile(os.path.join(checkpoint_folder, file)):
            if file.endswith('h5'):
                last_model_file = os.path.join(checkpoint_folder, file)
                break
    if last_model_file is None:
        glog.error('no files in ' + checkpoint_folder)
        return False
    glog.info('loading ' + last_model_file)
    self.load_model_from_file(file=last_model_file)
    glog.info('loading ' + last_model_file + ' ok')
    return True
  # do learning
  def train(self, x_train, y_train, x_val, y_val, checkpoint_folder=None):
    if checkpoint_folder is not None:
      self.load_last_checkpoint(checkpoint_folder=checkpoint_folder)
    tf.debugging.set_log_device_placement(True)
    # try to load from check point
    self.model.fit(x_train, y_train, batch_size=self.batch_size, validation_data=(x_val, y_val), epochs=self.epochs, workers=16, callbacks=self.get_callbacks())
    return True
  def predict(self, x_test):
      return self.model.predict(x_test)