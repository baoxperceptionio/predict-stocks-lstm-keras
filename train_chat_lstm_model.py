
# -*- coding: utf-8 -*-
# Brief: A Seq2Seq Model Implementation by keras Recurrent Neural Network LSTM.
# Author: Tateo_YANAGI @soarcloud.com
#
import numpy as np
import glog
import os
import pandas
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint

import msgpack_numpy as msg_np

import lstm_model

if __name__ == "__main__":
  glog.info("Num GPUs Available: " + str(len(tf.config.list_physical_devices('GPU'))))
  os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
  gpu_ids = ['0', '1']
  os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_ids)

  model_wrapper = lstm_model.LstmModel()
 
  # split data for learning and test by partition
  #x_train, y_train, x_val, y_val = model_wrapper.random_training_data()
  x_train, y_train, x_val, y_val = model_wrapper.load_training_data(htb_msgpack_file='./htb_vecs_1.msgpack', bth_msgpack_file='./bth_vecs_1.msgpack')
  # https://medium.com/@daniel820710/%E5%88%A9%E7%94%A8keras%E5%BB%BA%E6%A7%8Blstm%E6%A8%A1%E5%9E%8B-%E4%BB%A5stock-prediction-%E7%82%BA%E4%BE%8B-1-67456e0a0b
  # from 2 dimmension to 3 dimension
  #y_train = y_train[:, :, np.newaxis]
  #y_val = y_val[:, :]
  glog.info("x_train shape " + str(np.shape(x_train)))
  glog.info("y_train shape " + str(np.shape(y_train)))
  model_wrapper.create_model()
  # do learning
  checkpoint_folder = 'checkpoint'
  model_wrapper.train(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, checkpoint_folder=checkpoint_folder)
  # double check
  y_pred = model_wrapper.predict(x_train)
  distances = np.empty((y_pred.shape[0], 1))
  for it in range(y_pred.shape[0]):
    distances[it] = np.linalg.norm(y_pred[it, :] - y_train[it, :])
  result = pandas.DataFrame(distances)
  hist = result.hist()
  #result.columns = ['train pred l2 dis']
  #hist.plot()
  plt.show()
  plt.gcf().savefig('train_dis_hist.png')

  # test
  y_pred = model_wrapper.predict(x_val)
  glog.info("x_val shape" + str(np.shape(x_val)))
  glog.info("predicted shape " + str(np.shape(y_pred)) + " y_val.shape " + str(y_val.shape))
  distances = np.empty((y_pred.shape[0], 1))
  for it in range(y_val.shape[0]):
    distances[it] = np.linalg.norm(y_pred[it, :] - y_val[it, :])
  result = pandas.DataFrame(distances)
  hist = result.hist()
  plt.show()
  plt.gcf().savefig('val_dis_hist.png')

