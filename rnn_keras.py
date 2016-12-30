import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Bidirectional
from keras.layers import LSTM, GRU
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping
import math


class BatchGenerator(object):
    def __init__(self, isTraing, batch_size, target_col, feature_col, seq_len):
        if (isTraing):
            self.data = train_data
        else:
            self.data = test_data
        self.batch_size = batch_size
        self.data_flow = None
        # self.avg = train_data[:, column].mean()
        # self.std = train_data[:, column].std()
        # self.data = (self.data - self.avg) / self.std

        self.x, self.y = self.gru_split(target_col, feature_col, seq_len)
        self.max_len = self.x.shape[0]
        self.n_batch = self.max_len // self.batch_size
        self.cursor = 0
        self.reset()

    def reset(self):
        self.cursor = [self.n_batch * i for i in range(self.batch_size)]

    def get_all_pairs(self):
        return self.x, self.y

    def next(self):
        x_next = self.x[self.cursor]
        y_next = self.y[self.cursor]
        x_next = x_next.transpose(1, 0, 2)
        for i in xrange(batch_size):
            self.cursor[i] = (self.cursor[i] + 1) % self.max_len
        if self.cursor[0] >= self.batch_size:
            self.reset()
        return x_next, y_next

    def gru_split(self, target_col, features, seq_len):
        max_length = self.data.shape[0] - seq_len
        x = np.zeros([max_length, seq_len, len(features)])
        y = np.zeros([max_length, 1])
        for i in range(max_length):
            x[i, :, :] = self.data[i:i + seq_len, features]
            y[i, :] = self.data[i + seq_len, target_col]
        return x, y


# read data from .csv file and normalization
train_file = 'train_data.txt'
test_file = 'test_data.txt'
train_data = pd.read_csv(train_file, sep='\t', header=None).values
test_data = pd.read_csv(test_file, sep='\t', header=None).values
print 'Load data success!'
train_avg = train_data.mean(axis=0)
train_std = train_data.std(axis=0)
train_len = train_data.shape[0]
test_len = test_data.shape[0]
train_data = (train_data - np.tile(train_avg,[train_len,1]))/np.tile(train_std, [train_len,1])
test_data = (test_data - np.tile(train_avg, [test_len,1]))/np.tile(train_std, [test_len,1])
print 'normalization success!'
batch_size = 64
column = 0
feature_cols = [column]
seq_len = 6
def ProcessOneColumn(batch_size, column, feature_col, seq_len):
    train_set = BatchGenerator(True, batch_size=batch_size, target_col=column, feature_col=feature_col, seq_len=seq_len)
    test_set = BatchGenerator(False, batch_size=batch_size, target_col=column, feature_col=feature_col, seq_len=seq_len)
    X_train, y_train = train_set.get_all_pairs()
    X_test, y_test = test_set.get_all_pairs()
    return X_train, y_train, X_test, y_test
print 'Building data set successfully!'

def CreateModel(model_type, num_nerous=64, features=1, seq_len=6):
    model = Sequential()
    if(model_type=='LSTM'):
        # model.add(Bidirectional(LSTM(output_dim=num_nerous), input_shape=(seq_len,features)))
        model.add(LSTM(output_dim=num_nerous, input_dim=features, input_length=seq_len))
    else:
        # model.add(Bidirectional(GRU(output_dim=num_nerous), input_shape=(seq_len,features)))
        model.add(GRU(output_dim=num_nerous, input_dim=features, input_length=seq_len))
        # model.add(GRU(output_dim=num_nerous, input_shape=(seq_len, features)))
    # model.add(Dropout(0.5))
    # model.add(Activation('sigmoid'))
    model.add(Dense(1))
    # model.add(Activation('linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
#     model.fit(X_train, Y_train, nb_epoch=n_epoch, batch_size=batch_size)
    return model

def EvaluateColumn(model_type='LSTM',column=0, feature_col=[0], mape_const=0.0, rmse_const=1.0, n_epoch=1, batch_size=64, seq_len=6):
    # Invert predictions
    X_train, y_train, X_test, y_test = ProcessOneColumn(batch_size=batch_size, column=column, feature_col=feature_col, seq_len=seq_len)
    model = CreateModel(model_type, num_nerous=64, features=len(feature_col), seq_len=seq_len)
    # early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    # model.fit(X_train, y_train, nb_epoch=n_epoch, batch_size=batch_size,validation_split=0.1,callbacks=[early_stopping], shuffle=True)

    # set verbose=0 to hide the verbose training process
    model.fit(X_train, y_train, nb_epoch=n_epoch, batch_size=batch_size, shuffle=True, verbose=0)
    y_predict = model.predict(X_test)

    def mean_absolute_percentage_error(y_true, y_pred):
        return np.mean(np.abs(1.0 * (y_true - y_pred) / (y_true + mape_const)), dtype=np.float32)

    RMSE = math.sqrt(mean_squared_error(y_test, y_predict)) * rmse_const
    MAPE = mean_absolute_percentage_error(y_test, y_predict)
    MSE = np.square(RMSE)
    print('Test Score: %.5f MSE' % (RMSE * RMSE))
    print('Test Score: %.5f MAPE' % (MAPE))
    return MSE,MAPE
relations = pd.read_csv('causality.txt',dtype='str',sep='\t',header=None)

seq_len = 6
result = np.zeros([50,8])
for cols in xrange(50):
    # feature_cols = relations.ix[cols, 0].split(',')
    print 'start column {}, single point'.format(cols)
    feature_cols = [cols]
    rmse_const = train_std[cols]
    mape_const = train_avg[cols]/train_std[cols]

    MSE,MAPE = EvaluateColumn('LSTM', cols, feature_cols,mape_const,rmse_const,n_epoch=25, seq_len=seq_len)
    result[cols,0] = MSE
    result[cols,1] = MAPE

    MSE, MAPE = EvaluateColumn('GRU', cols, feature_cols, mape_const, rmse_const, n_epoch=25, seq_len=seq_len)
    result[cols,2] = MSE
    result[cols,3] = MAPE

    feature_cols = relations.ix[cols, 0].split(',')
    print 'start column {}, multi point'.format(cols)
    MSE,MAPE = EvaluateColumn('LSTM', cols, feature_cols,mape_const,rmse_const,n_epoch=25, seq_len=seq_len)
    result[cols,4] = MSE
    result[cols,5] = MAPE

    MSE, MAPE = EvaluateColumn('GRU', cols, feature_cols, mape_const, rmse_const, n_epoch=25, seq_len=seq_len)
    result[cols,6] = MSE
    result[cols,7] = MAPE
np.savetxt('result_seq_6_single_and_multi.csv',result,fmt='%6.3f',delimiter=',',newline='\n',header='MSE(LSTM),MAPE(LSTM),MSE(GRU),MAPE(GRU),MSE(LSTM),MAPE(LSTM),MSE(GRU),MAPE(GRU)',footer='',comments='')
print 'finish the situation seq=6, congratulations!'
seq_len = 30
result = np.zeros([50,8])
for cols in xrange(50):
    # feature_cols = relations.ix[cols, 0].split(',')
    print 'start column {}, single point'.format(cols)
    feature_cols = [cols]
    rmse_const = train_std[cols]
    mape_const = train_avg[cols]/train_std[cols]

    MSE,MAPE = EvaluateColumn('LSTM', cols, feature_cols,mape_const,rmse_const,n_epoch=25, seq_len=seq_len)
    result[cols,0] = MSE
    result[cols,1] = MAPE

    MSE, MAPE = EvaluateColumn('GRU', cols, feature_cols, mape_const, rmse_const, n_epoch=25, seq_len=seq_len)
    result[cols,2] = MSE
    result[cols,3] = MAPE

    feature_cols = relations.ix[cols, 0].split(',')
    print 'start column {}, multi point'.format(cols)
    MSE,MAPE = EvaluateColumn('LSTM', cols, feature_cols,mape_const,rmse_const,n_epoch=25, seq_len=seq_len)
    result[cols,4] = MSE
    result[cols,5] = MAPE

    MSE, MAPE = EvaluateColumn('GRU', cols, feature_cols, mape_const, rmse_const, n_epoch=25, seq_len=seq_len)
    result[cols,6] = MSE
    result[cols,7] = MAPE
np.savetxt('result_seq_30_single_and_multi.csv',result,fmt='%6.3f',delimiter=',',newline='\n',header='MSE(LSTM),MAPE(LSTM),MSE(GRU),MAPE(GRU),MSE(LSTM),MAPE(LSTM),MSE(GRU),MAPE(GRU)',footer='',comments='')
print 'finish the situation seq=30, congratulations!'
