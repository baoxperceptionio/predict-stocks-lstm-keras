
# -*- coding: utf-8 -*-
# Brief: A Seq2Seq Model Implementation by keras Recurrent Neural Network LSTM.
# Author: Tateo_YANAGI @soarcloud.com
#
import numpy
import glog
import pandas
import matplotlib.pyplot as plt
 
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
 
class Predict:
  def __init__(self):
    self.length_of_sequences = 5
    self.bert_dim = 648
    self.batch_size = 1
    self.train_sample_num = 995
    self.epochs = 100
    self.percentage = 0.8
 
  # prepare data
  def load_data(self, data, n_prev):
    x, y = [], []
    for i in range(len(data) - n_prev):
      x.append(data.iloc[i:(i+n_prev)].values)
      y.append(data.iloc[i+n_prev].values)
    X = numpy.array(x)
    Y = numpy.array(y)
    return X, Y

  # stress test
  def random_data(self, seq_num):
    X = numpy.random.rand(seq_num, self.length_of_sequences, self.bert_dim)
    Y = numpy.random.rand(seq_num, self.length_of_sequences, self.bert_dim)
    return X, Y

 
  # make model
  def create_model(self) :
    Model = Sequential()
    # Expected input batch shape: (batch_size, timesteps, data_dim).
    input_shape=(self.length_of_sequences, self.bert_dim)
    batch_input_shape=(self.batch_size, self.length_of_sequences, self.bert_dim)
    # units: Positive integer, dimensionality of the output space 
    Model.add(LSTM(units=self.bert_dim, input_shape=input_shape, return_sequences=False))
    Model.add(Dense(self.bert_dim))
    Model.add(Activation("linear"))
    Model.compile(loss="mape", optimizer="adam")
    return Model
 
  # do learning
  def train(self, x_train, y_train) :
    Model = self.create_model()
    glog.info(Model.summary())
    Model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs)
    return Model
 
if __name__ == "__main__":
 
  predict = Predict()
  nstocks = 2
 
  # do learning, prediction, show to each stock
  for istock in range(1, nstocks + 1):
    
    # prepare data 
    data = None
    data = pandas.read_csv('./csv/' + str(istock) + '_stock_price.csv')
    data.columns = ['date', 'open', 'high', 'low', 'close']
    data['date'] = pandas.to_datetime(data['date'], format='%Y-%m-%d')
 
    # standalize the close data
    data['close'] = preprocessing.scale(data['close'])
    data = data.sort_values(by='date')
    data = data.reset_index(drop=True)
    data = data.loc[:, ['date', 'close']]
 
    # split data for learning and test by partition
    split_pos = int(len(data) * predict.percentage)
    #x_train, y_train = predict.load_data(data[['close']].iloc[0:split_pos], predict.length_of_sequences)
    x_train, y_train = predict.random_data(seq_num=self.train_sample_num)
    # x_test, y_test = predict.load_data(data[['close']].iloc[split_pos:], predict.length_of_sequences)
    glog.info("x_train shape " + str(numpy.shape(x_train)))
    glog.info("y_train shape " + str(numpy.shape(y_train)))
    # do learning
    
    model = predict.train(x_train, y_train)
    # do test
    x_test, y_test = predict.load_data(data[['close']].iloc[split_pos:], predict.length_of_sequences)
    predicted = model.predict(x_test)
    glog.info("predicted shape " + str(numpy.shape(predicted)))
    result = pandas.DataFrame(predicted)
    result.columns = [str(istock) + '_predict']
    result[str(istock) + '_actual'] = y_test
 
    # show
    result.plot()
    plt.show()
 
    # compare prices
    current = result.iloc[-1][str(istock) + '_actual']
    predictable = result.iloc[-1][str(istock) + '_predict']
    if (predictable - current) > 0:
      print(f'{istock} stock price of the next day INcreases: {predictable-current:.2f}, predictable:{predictable:.2f}, current:{current:.2f}')
    else:
      print(f'{istock} stock price of the next day DEcreases: {current-predictable:.2f}, predictable:{predictable:.2f}, current:{current:.2f}')
