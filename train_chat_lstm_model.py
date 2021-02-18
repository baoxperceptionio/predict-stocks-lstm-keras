
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
  #gpu_ids = ['0']
  #os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_ids)
  round = '2'
  checkpoint_folder = 'checkpoint_rnd'+round
  if not os.path.exists(checkpoint_folder):
    os.mkdir(checkpoint_folder)
  model_wrapper = lstm_model.LstmModel(gpu_devices=["GPU:0"], output_dir=checkpoint_folder)
  x_train, y_train, x_val, y_val = model_wrapper.load_training_data(htb_msgpack_file='./htb_vecs_' + round + '.msgpack',
                                                                    bth_msgpack_file='./bth_vecs_' + round + '.msgpack')
  glog.info("x_train shape " + str(np.shape(x_train)))
  glog.info("y_train shape " + str(np.shape(y_train)))
  model_wrapper.create_model()
  
  final_h5_out_path = os.path.join(checkpoint_folder, round + '_final.h5')
  if os.path.isfile(final_h5_out_path):
    glog.info('loading ' + final_h5_out_path)
    model_wrapper.load_model_from_file(final_h5_out_path)
  # do learning
  if True:
    model_wrapper.epochs = 100000
    model_wrapper.train(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, load_checkpoint_folder=None)
    model_wrapper.model.save_weights(final_h5_out_path)
  # double check
  y_pred = model_wrapper.predict(x_train)
  glog.info("y_pred.shape" + str(y_pred.shape))
  glog.info("y_train.shape" + str(y_train.shape))
  glog.info("x_train.shape" + str(x_train.shape))
  for it in range(1,10):
    y_train[it, 0, :] =  y_train[it, 0, :] / np.linalg.norm(y_train[it, 0, :])
    glog.info('train ' + str(y_train[it, 0, 1:10]))
    y_pred[it, :] =  y_pred[it, :] / np.linalg.norm(y_pred[it, :])
    glog.info('pred ' + str(y_pred[it, 1:10]))
    loss = tf.keras.losses.cosine_similarity(y_pred[it, :], y_train[it, 0, :], axis=1)
    glog.info('loss ' + str(loss))
    dot_prod = np.dot(y_pred[it, :], y_train[it, 0, :]) / np.linalg.norm(y_pred[it, :]) / np.linalg.norm(y_train[it, 0, :])
    glog.info('dot_prod ' + str(dot_prod))
    dp = 0
    for it_dim in range(y_pred.shape[1]):
      dp = dp + y_pred[it, it_dim] * y_train[it, 0, it_dim]
    dp = dp / np.linalg.norm(y_pred[it, :]) / np.linalg.norm(y_train[it, 0, :])
    glog.info("dp=" + str(dp))

  cos_loss = np.empty((y_pred.shape[0], 1))
  distances = np.empty((y_pred.shape[0], 1))
  for it in range(y_pred.shape[0]):
    cos_loss[it] = tf.keras.losses.cosine_similarity(y_pred[it, :], y_train[it, :], axis=1)
    distances[it] = np.linalg.norm(y_pred[it, :] - y_train[it, :])
  df = pandas.DataFrame(distances)
  hist = df.hist()
  plt.show()
  plt.gcf().savefig(round + '_train_dis_hist.png')
  df = pandas.DataFrame(cos_loss)
  hist = df.hist()
  plt.show()
  plt.gcf().savefig(round + '_train_cos_loss_hist.png')

  # test
  y_pred = model_wrapper.predict(x_val)
  for it in range(1,10):
    glog.info(str(y_pred[it, 1:10]))
  glog.info("x_val shape" + str(np.shape(x_val)))
  glog.info("predicted shape " + str(np.shape(y_pred)) + " y_val.shape " + str(y_val.shape))
  distances = np.empty((y_pred.shape[0], 1))
  for it in range(y_val.shape[0]):
    distances[it] = np.linalg.norm(y_pred[it, :] - y_val[it, :])
  result = pandas.DataFrame(distances)
  hist = result.hist()
  plt.show()
  plt.gcf().savefig(round + '_val_dis_hist.png')

