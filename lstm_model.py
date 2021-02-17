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
from keras.layers import Reshape
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint

import msgpack_numpy as msg_np

class LstmModel:
  def __init__(self, epochs=30000, bert_dim=768, output_dir='checkpoint', gpu_devices=["GPU:0", "GPU:1"]):
    self.length_of_sequences = -1
    self.bert_dim = bert_dim
    self.batch_size = 1024
    self.train_sample_num = 0
    self.epochs = epochs
    self.output_dir = output_dir
    self.gpu_devices = gpu_devices
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
      monitor='loss', min_delta=1e-4, patience=100, verbose=0,
      mode='auto', baseline=None, restore_best_weights=True)

    callbacks = [model_checkpoint, early_stop]

    return callbacks
  def check_training_val_data_format(self, x_train, y_train, x_val, y_val):
    if (x_train.ndim != 3) or (x_train.shape[1] != self.length_of_sequences * 2 - 1):
      glog.error('weird x_train shape ' + str(x_train.shape))
      return False
    if (x_val.ndim != 3) or (x_val.shape[1] != self.length_of_sequences * 2 - 1):
      glog.error('x_val shape ' + str(x_val.shape))
      return False
    if y_train.ndim != 3 or y_train.shape[1] != 1:
      glog.fatal('werid y_train.shape ' + str(y_train.shape))
      return False
    if y_val.ndim != 3 or y_val.shape[1] != 1:
      glog.fatal('werid y_val.ndim ' + str(y_val.shape))
      return False
    for it in range(y_train.shape[0]):
      norm = np.linalg.norm(y_train[it, 0, :])
      if norm > 1 + 1e-4 or norm < 1 - 1e-4:
        glog.error('werid y_train norm=' + str(norm))
    for it in range(y_val.shape[0]):
      norm = np.linalg.norm(y_val[it, 0, :])
      if norm > 1 + 1e-4 or norm < 1 - 1e-4:
        glog.error('werid y_val norm=' + str(norm))
    return True
  # prepare data
  def load_training_data(self, htb_msgpack_file, bth_msgpack_file):
    with open(htb_msgpack_file, 'rb') as f:
      htb_vecs = msg_np.unpackb(f.read())
      for it in range(htb_vecs.shape[0]):
        if htb_vecs.ndim == 2:
          norm = np.linalg.norm(htb_vecs[it, :])
          if norm > 1 + 1e-4 or norm < 1 - 1e-4:
            for it_col in range(htb_vecs.shape[1]):
              htb_vecs[it, it_col] = htb_vecs[it, it_col] / norm
            glog.error(str(it) + ' werid htb_vecs norm=' + str(norm))
        else:
          for it_msg in range(htb_vecs.shape[1]):
            norm = np.linalg.norm(htb_vecs[it, it_msg, :])
            if norm > 1 + 1e-4 or norm < 1 - 1e-4:
              for it_col in range(htb_vecs.shape[2]):
                htb_vecs[it, it_msg, it_col] = htb_vecs[it, it_msg, it_col] / norm
              glog.error(str(it) + ' werid htb_vecs norm=' + str(norm))

    with open(bth_msgpack_file, 'rb') as f:
      bth_vecs = msg_np.unpackb(f.read())
      for it in range(bth_vecs.shape[0]):
        if bth_vecs.ndim == 2:
          norm = np.linalg.norm(bth_vecs[it, :])
          if norm > 1 + 1e-4 or norm < 1 - 1e-4:
            for it_col in range(bth_vecs.shape[1]):
              bth_vecs[it, it_col] = bth_vecs[it, it_col] / norm
            glog.error(str(it) + ' werid bth_vecs norm=' + str(norm))
        else:
          for it_msg in range(bth_vecs.shape[1]):
            norm = np.linalg.norm(bth_vecs[it, it_msg, :])
            if norm > 1 + 1e-4 or norm < 1 - 1e-4:
              for it_col in range(bth_vecs.shape[2]):
                bth_vecs[it, it_msg, it_col] = bth_vecs[it, it_msg, it_col] / norm
              glog.error(str(it) + ' werid bth_vecs norm=' + str(norm))
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
    randomList = np.arange(self.train_sample_num)
    np.random.shuffle(randomList)
    htb_vecs = htb_vecs[randomList, :, :]
    bth_vecs = bth_vecs[randomList, :, :]
    # given hhhh, bbbb: hbhbhbh is x, b is y
    x = np.empty((self.train_sample_num, self.length_of_sequences * 2 - 1, self.bert_dim))
    y = np.empty((self.train_sample_num, 1, self.bert_dim))
    for it_sample in range(self.train_sample_num):
      for it_round in range(self.length_of_sequences):
        x[it_sample, it_round * 2, :] = htb_vecs[it_sample, it_round, :]
        if it_round < self.length_of_sequences - 1:
          x[it_sample, it_round * 2 + 1, :] = bth_vecs[it_sample, it_round, :]
      y[it_sample, 0, :] = bth_vecs[it_sample, -1, :]
    val_percetage = 0.05
    split_pos = int(self.train_sample_num * val_percetage)
    x_train = x[split_pos:, :, :]
    y_train = y[split_pos:, :, :]
    if y_train.ndim == 3:
      y_train = y_train[:, -1, :]
    if y_train.ndim == 2:
      y_train = y_train[:, np.newaxis, :]
    x_val = x[:split_pos, :, :]
    y_val = y[:split_pos, :, :]
    if y_val.ndim == 3:
      y_val = y_val[:, -1, :]
    if y_val.ndim == 2:
      y_val = y_val[:, np.newaxis, :]
    if not self.check_training_val_data_format(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val):
      exit()
    return x_train, y_train, x_val, y_val

  # stress test
  def random_training_data(self):
    # generate random perfect data for testing or for boostrap model parameters
    if self.length_of_sequences < 0:
      self.length_of_sequences = 5
    self.train_sample_num = 10000
    x_train = np.random.rand(self.train_sample_num, self.length_of_sequences * 2 - 1, self.bert_dim)
    for it0 in range(x_train.shape[0]):
      for it1 in range(x_train.shape[1]):
        x_train[it0, it1, :] = x_train[it0, it1, :] / np.linalg.norm(x_train[it0, it1, :])
    y_train = x_train[:, -1, :] # so we can get a perfect fit
    y_train = y_train[:, np.newaxis, :]
    #y_train = np.random.rand(y_train.shape[0], y_train.shape[1], y_train.shape[2])
    for it0 in range(y_train.shape[0]):
      y_train[it0, 0, :] = y_train[it0, 0, :] / np.linalg.norm(y_train[it0, 0, :])
    x_val = np.random.rand(int(self.train_sample_num * 0.1), self.length_of_sequences * 2 - 1, self.bert_dim)
    y_val = x_val[:, -1, :] # so we can get a perfect fit
    y_val = y_val[:, np.newaxis, :]
    #y_val = np.random.rand(y_val.shape[0], y_val.shape[1], y_val.shape[2])
    for it0 in range(y_val.shape[0]):
      y_val[it0, 0, :] = y_val[it0, 0, :] / np.linalg.norm(y_val[it0, 0, :])
    if not self.check_training_val_data_format(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val):
      exit()
    return x_train, y_train, x_val, y_val
 
  # make model
  def create_model(self):
    if self.length_of_sequences == 0:
        glog.error("self.length_of_sequences = 0. you need to load data first (or use random_data())")
        return None
    strategy = tf.distribute.MirroredStrategy(devices=self.gpu_devices)
    with strategy.scope():
      self.model = Sequential()
      lstm_out_units = int(self.bert_dim / 8) # this does not neccessarily equal to bert dim
      # see the # of param calculation in https://datascience.stackexchange.com/questions/10615/number-of-parameters-in-an-lstm-model
      # https://arxiv.org/pdf/1409.3215.pdf 
      # We found deep LSTMs to significantly outperform shallow LSTMs, 
      # where each additional layer reduced perplexity by nearly 10%, possibly due to their much larger hidden state. 
      self.model.add(LSTM(units=lstm_out_units, input_length=self.length_of_sequences * 2 - 1, input_dim=self.bert_dim, return_sequences=False, dropout=0.1))
      self.model.add(Reshape(target_shape=(1, lstm_out_units)))
      self.model.add(LSTM(units=lstm_out_units, input_length=1, input_dim=lstm_out_units, return_sequences=False, dropout=0.1))
      self.model.add(Reshape(target_shape=(1, lstm_out_units)))
      self.model.add(LSTM(units=lstm_out_units, input_length=1, input_dim=lstm_out_units, return_sequences=False, dropout=0.1))
      self.model.add(Reshape(target_shape=(1, lstm_out_units)))
      self.model.add(LSTM(units=lstm_out_units, input_length=1, input_dim=lstm_out_units, return_sequences=False, dropout=0.1))
      self.model.add(Dropout(rate=0.1))
      self.model.add(Dense(units=self.bert_dim, activation='linear'))
      # cosine_similarity mean_squared_error
      self.model.compile(loss="cosine_similarity", optimizer="sgd")
      self.model.summary()
      return True
    glog.error('multigpu setting failed')
    return False
  def load_model_from_file(self, file):
    if self.model is None:
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
    # try to load from check point
    if checkpoint_folder is not None:
      self.load_last_checkpoint(checkpoint_folder=checkpoint_folder)
    # tf.debugging.set_log_device_placement(True)
    self.model.fit(x_train, y_train, batch_size=self.batch_size, validation_data=(x_val, y_val), epochs=self.epochs, workers=16, callbacks=self.get_callbacks())
    return True
  def predict(self, x_test):
      return self.model.predict(x_test)