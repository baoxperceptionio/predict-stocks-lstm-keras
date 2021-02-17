# -*- coding: utf-8 -*-
# Brief: A Seq2Seq Model Implementation by keras Recurrent Neural Network LSTM.
# Author: Tateo_YANAGI @soarcloud.com
#
import numpy as np
import glog
import os
import tensorflow as tf
import unittest
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
      tf.cast(htb_vecs, dtype=tf.float32)
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
      tf.cast(bth_vecs, dtype=tf.float32)
    self.length_of_sequences = np.shape(htb_vecs)[1]
    train_sample_num = np.shape(htb_vecs)[0]
    self.bert_dim = np.shape(htb_vecs)[2]
    glog.info('self.length_of_sequences=' + str(self.length_of_sequences))
    glog.info('train_sample_num=' + str(train_sample_num))
    glog.info('self.bert_dim=' + str(self.bert_dim))
    glog.check_eq(self.length_of_sequences, np.shape(bth_vecs)[1])
    glog.check_eq(train_sample_num, np.shape(bth_vecs)[0])
    glog.check_eq(self.bert_dim, np.shape(bth_vecs)[2])
    # split data
    # shuffle
    randomList = np.arange(train_sample_num)
    np.random.shuffle(randomList)
    htb_vecs = htb_vecs[randomList, :, :]
    bth_vecs = bth_vecs[randomList, :, :]
    # given hhhh, bbbb: hbhbhbh is x, b is y
    x = np.empty((train_sample_num, self.length_of_sequences * 2 - 1, self.bert_dim))
    y = np.empty((train_sample_num, 1, self.bert_dim))
    for it_sample in range(train_sample_num):
      for it_round in range(self.length_of_sequences):
        x[it_sample, it_round * 2, :] = htb_vecs[it_sample, it_round, :]
        if it_round < self.length_of_sequences - 1:
          x[it_sample, it_round * 2 + 1, :] = bth_vecs[it_sample, it_round, :]
      y[it_sample, 0, :] = bth_vecs[it_sample, -1, :]
    val_percetage = 0.05
    split_pos = int(train_sample_num * val_percetage)
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
  def random_training_data(self, train_sample_num):
    # generate random perfect data for testing or for boostrap model parameters
    if self.length_of_sequences < 0:
      self.length_of_sequences = 5
    x_train = np.random.rand(train_sample_num, self.length_of_sequences * 2 - 1, self.bert_dim).astype(float)
    for it0 in range(x_train.shape[0]):
      for it1 in range(x_train.shape[1]):
        x_train[it0, it1, :] = x_train[it0, it1, :] / np.linalg.norm(x_train[it0, it1, :])
    y_train = x_train[:, -1, :] # so we can get a perfect fit
    if False:
      y_train = y_train[:, np.newaxis, :]
      #y_train = np.random.rand(y_train.shape[0], y_train.shape[1], y_train.shape[2])
      for it0 in range(y_train.shape[0]):
        y_train[it0, 0, :] = y_train[it0, 0, :] / np.linalg.norm(y_train[it0, 0, :])
    else:
      #y_train = np.random.rand(y_train.shape[0], y_train.shape[1], y_train.shape[2])
      for it0 in range(y_train.shape[0]):
        y_train[it0, :] = y_train[it0, :] / np.linalg.norm(y_train[it0, :])

    x_val = np.random.rand(int(train_sample_num * 0.1), self.length_of_sequences * 2 - 1, self.bert_dim).astype(float)
    y_val = x_val[:, -1, :] # so we can get a perfect fit
    if False:
      y_val = y_val[:, np.newaxis, :]
      #y_val = np.random.rand(y_val.shape[0], y_val.shape[1], y_val.shape[2])
      for it0 in range(y_val.shape[0]):
        y_val[it0, 0, :] = y_val[it0, 0, :] / np.linalg.norm(y_val[it0, 0, :])
    else:
      #y_val = np.random.rand(y_val.shape[0], y_val.shape[1], y_val.shape[2])
      for it0 in range(y_val.shape[0]):
        y_val[it0, :] = y_val[it0, :] / np.linalg.norm(y_val[it0, :])

    if not self.check_training_val_data_format(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val):
      pass#exit()
    return x_train, y_train, x_val, y_val
   
  def random_training_dataset(self, sample_num=100):
    # generate random perfect data for testing or for boostrap model parameters
    if self.length_of_sequences < 0:
      self.length_of_sequences = 5
    train_x = []
    train_y = []
    for it in range(sample_num):
      x = np.random.rand(self.length_of_sequences * 2 - 1, self.bert_dim).astype(float)
      for it1 in range(x.shape[0]):
        x[it1, :] = x[it1, :] / np.linalg.norm(x[it1, :])
      y = x[-1, :]
      x = x[np.newaxis, :, :]
      array_sum = np.sum(x)
      if np.isnan(array_sum):
        glog.fatal('x nan')
      y = y[np.newaxis, np.newaxis, :]
      train_x.append(x)
      train_y.append(y)
    val_x = []
    val_y = []
    for it in range(int(sample_num * 0.05)):
      x = np.random.rand(self.length_of_sequences * 2 - 1, self.bert_dim).astype(float)
      for it1 in range(x.shape[0]):
        x[it1, :] = x[it1, :] / np.linalg.norm(x[it1, :])
      x = x[np.newaxis, :, :]
      array_sum = np.sum(x)
      if np.isnan(array_sum):
        glog.fatal('x nan')
      y = x[-1, :]
      y = y[np.newaxis, np.newaxis, :]
      val_x.append(x)
      val_y.append(y)

    #if not self.check_training_val_data_format(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val):
    #  exit()
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    train_dataset = train_dataset.with_options(options)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y))
    val_dataset = train_dataset.with_options(options)

    return train_dataset, val_dataset
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
      timestamp_num = self.length_of_sequences * 2 - 1
      self.model.add(LSTM(units=lstm_out_units, input_length=timestamp_num, input_dim=self.bert_dim, return_sequences=False, dropout=0.1))
      self.model.add(Reshape(target_shape=(1, lstm_out_units)))
      self.model.add(LSTM(units=lstm_out_units, input_length=1, input_dim=lstm_out_units, return_sequences=False, dropout=0.1))
      self.model.add(Reshape(target_shape=(1, lstm_out_units)))
      self.model.add(LSTM(units=lstm_out_units, input_length=1, input_dim=lstm_out_units, return_sequences=False, dropout=0.1))
      self.model.add(Reshape(target_shape=(1, lstm_out_units)))
      self.model.add(LSTM(units=lstm_out_units, input_length=1, input_dim=lstm_out_units, return_sequences=False, dropout=0.1))
      self.model.add(Dropout(rate=0.1))
      self.model.add(Dense(units=self.bert_dim, activation='linear'))
      # cosine_similarity mean_squared_error
      self.model.compile(loss=tf.keras.losses.CosineSimilarity(axis=-1), optimizer="sgd")
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
  def train(self, x_train=None, y_train=None, x_val=None, y_val=None, train_dataset=None, val_dataset=None, checkpoint_folder=None):
    # try to load from check point
    if checkpoint_folder is not None:
      self.load_last_checkpoint(checkpoint_folder=checkpoint_folder)
    if train_dataset is not None:
      self.model.fit(train_dataset, validation_data=val_dataset, batch_size=self.batch_size, epochs=self.epochs, workers=16, callbacks=self.get_callbacks())
    else:
      # tf.debugging.set_log_device_placement(True)
      self.model.fit(x_train, y_train, batch_size=self.batch_size, validation_data=(x_val, y_val), epochs=self.epochs, workers=16, callbacks=self.get_callbacks())
    return True
  def predict(self, x_test):
      return self.model.predict(x_test)

class TestLSTM(unittest.TestCase):
  def test_random_train(self):
    model_wrapper = LstmModel()
    train_ds, val_ds = model_wrapper.random_training_dataset(sample_num=10000)
    model_wrapper.create_model()
    model_wrapper.epochs = 10
    for elem in train_ds:
      self.assertEqual(elem[0].numpy().shape[0], 1)
      self.assertEqual(elem[0].numpy().shape[1], model_wrapper.length_of_sequences * 2 - 1)
      self.assertEqual(elem[0].numpy().shape[2], model_wrapper.bert_dim)
      self.assertEqual(elem[1].numpy().shape[0], 1)
      self.assertEqual(elem[1].numpy().shape[1], 1)
      self.assertEqual(elem[0].numpy().shape[2], model_wrapper.bert_dim)

    for elem in val_ds:
      self.assertEqual(elem[0].numpy().shape[0], 1)
      self.assertEqual(elem[0].numpy().shape[1], model_wrapper.length_of_sequences * 2 - 1)
      self.assertEqual(elem[0].numpy().shape[2], model_wrapper.bert_dim)
      self.assertEqual(elem[1].numpy().shape[0], 1)
      self.assertEqual(elem[1].numpy().shape[1], 1)
      self.assertEqual(elem[0].numpy().shape[2], model_wrapper.bert_dim)
    model_wrapper.train(train_dataset=train_ds, val_dataset=val_ds, checkpoint_folder=None)

class TestNumpy(unittest.TestCase):
  def test_loss(self):
    y1 = np.array([1, 2, 3, 4, 5, 6], dtype=np.single)
    y2 = np.array([1, 2, 3, 4, 5, 6], dtype=np.single)
    self.assertEqual(y1.shape[0], 6)
    self.assertEqual(len(y1.shape), 1)
    self.assertEqual(y2.shape[0], 6)
    self.assertEqual(len(y2.shape), 1)
    y = [y1, y2]
    for it in range(2):
      y[it] = y[it] / np.linalg.norm(y[it])
    # validate normalization
    for it in range(2):
      sum = 0
      for it_col in range(y[0].shape[0]):
        sum = sum + y[0][it_col] * y[0][it_col]
      self.assertAlmostEqual(sum, 1, 5)
    # make them 2d
    y[0] = y[0][np.newaxis, :]
    y[1] = y[1][np.newaxis, :]
    loss = tf.keras.losses.cosine_similarity(tf.cast(y[0], dtype=tf.float32), tf.cast(y[1], dtype=tf.float32), axis=-1)
    self.assertEqual(loss.shape[0], 1)
    glog.info('test_loss loss shape ' + str(loss.shape))
    self.assertAlmostEqual(float(loss[0]), -1, 5)
    loss = tf.keras.losses.cosine_similarity([tf.cast(y[0], dtype=tf.float32)], [tf.cast(y[1], dtype=tf.float32)], axis=-1)
    glog.info('test_loss loss shape ' + str(loss.shape))
    self.assertEqual(loss.shape[0], 1)
    self.assertAlmostEqual(float(loss[0]), -1, 5)
    # make y[0] 3d
    y[0] = y[0][np.newaxis, np.newaxis, :]
    loss = tf.keras.losses.cosine_similarity(tf.cast(y[0], dtype=tf.float32), tf.cast(y[1], dtype=tf.float32), axis=-1)
    self.assertEqual(loss.shape[0], 1)
    glog.info('test_loss loss shape ' + str(loss.shape))
    # self.assertAlmostEqual(float(loss[0]), -1, 5)
  
  def test_cosine_similarity(self):
    sample_num = 10
    data_dim = 3
    for sample_num in [1, 10, 100]:
      y1 = np.random.rand(sample_num, data_dim).astype(float)
      y2 = np.random.rand(sample_num, data_dim).astype(float)
      loss = tf.keras.losses.cosine_similarity(y1, y2, axis=-1)
      glog.info('test_cosine_similarity loss shape axis=-1 ' + str(loss.shape))
      self.assertEqual(loss.shape[0], sample_num)
      loss = tf.keras.losses.cosine_similarity(y1, y2, axis=0)
      glog.info('test_cosine_similarity loss shape axis=0 ' + str(loss.shape))
      self.assertEqual(loss.shape[0], data_dim)
      loss = tf.keras.losses.cosine_similarity(y1, y2, axis=1)
      glog.info('test_cosine_similarity loss shape axis=1 ' + str(loss.shape))
      self.assertEqual(loss.shape[0], sample_num)
      for it in range(sample_num):
        dp = 0
        for it_col in range(data_dim):
          dp = dp + y1[it, it_col] * y2[it, it_col]
        dp = dp / np.linalg.norm(y1[it, :]) / np.linalg.norm(y2[it, :])
        loss2 = tf.keras.losses.cosine_similarity([y1[it,:]], [y2[it, :]], axis=1)
        self.assertAlmostEqual(float(loss2), -dp, 5)


  def test_loss2_cpu(self):
    sample_num = 3
    data_dim = 2
    y1 = np.random.rand(sample_num, 1, data_dim).astype(float)
    y2 = np.random.rand(sample_num, data_dim).astype(float)
    for it in range(sample_num):
      y1[it, 0, :] = y1[it, 0, :] / np.linalg.norm(y1[it, 0, :])
      y2[it, :] = y2[it, :] / np.linalg.norm(y2[it, :])
    # validate normalization
    for it in range(sample_num):
      sum1 = 0
      sum2 = 0
      for it_col in range(data_dim):
        sum1 = sum1 + y1[it, 0, it_col] * y1[it, 0, it_col]
        sum2 = sum2 + y2[it, it_col] * y2[it, it_col]
      self.assertAlmostEqual(sum1, 1, 5)
      self.assertAlmostEqual(sum2, 1, 5)
    # validate loss 1 by 1
    for it in range(sample_num):
      dp = 0
      for it_col in range(data_dim):
        dp = dp + y1[it, 0, it_col] * y2[it, it_col]
      # In the case of row access, the empty slice can be omitted for a more compact syntax:
      loss = tf.keras.losses.cosine_similarity([y1[it, 0, :]], [y2[it, :]], axis=1)
      self.assertEqual(len(loss.shape), 1)
      self.assertEqual(loss.shape[0], 1)
      self.assertAlmostEqual(float(loss[0]), -dp, 5)



if __name__ == "__main__":
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
  gpu_ids = ['0', '1']
  os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_ids)
  #os.environ["CUDA_VISIBLE_DEVICES"] = ""
  gpu_devices = tf.config.experimental.list_physical_devices('GPU')
  for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
  unittest.main(TestNumpy())
