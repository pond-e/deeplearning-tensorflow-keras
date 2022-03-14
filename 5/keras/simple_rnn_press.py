import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import SimpleRNN
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import csv

with open('./press_log_100Hz/press_logs_20220309-102401_2_checked.csv') as f:
    reader = csv.reader(f)
    l = [row for row in reader]

f = [k[1:] for k in l[1:]]
for i in range(len(f)):
    for j in range(len(f[0])):
        f[i][j] = float(f[i][j])
f = np.array(f)


np.random.seed(0)

def sin(x, T=100):
    return np.sin(2.0 * np.pi * x / T)


def toy_problem(T=100, ampl=0.05):
    x = np.arange(0, 2 * T + 1)
    noise = ampl * np.random.uniform(low=-1.0, high=1.0, size=len(x))
    return sin(x) + noise

'''
データの生成
'''
noise = np.random.uniform(low=-30, high=30, size=(len(f), 3))
l = f+noise
I = 3

length_of_sequences = len(l)
maxlen = 25  # ひとつの時系列データの長さ

data = np.zeros((length_of_sequences - maxlen + 1, maxlen, I))
target = np.zeros((length_of_sequences - maxlen + 1, I))

for i in range(0, length_of_sequences - maxlen ):
    data[i] = l[i: i + maxlen]
    target[i] = l[i + maxlen]

X = np.array(data).reshape(len(data), maxlen, I)
Y = np.array(target).reshape(len(data), I)

# データ設定
N_train = int(len(data) * 0.9)
N_validation = len(data) - N_train

X_train, X_validation, Y_train, Y_validation = \
    train_test_split(X, Y, test_size=N_validation)

'''
モデル設定
'''
n_in = len(X[0][0])  # 1
n_hidden = 20
n_out = len(Y[0])  # 1


def weight_variable(shape, name=None):
    return np.random.normal(scale=.01, size=shape)


early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

model = Sequential()
model.add(SimpleRNN(n_hidden,
                    kernel_initializer=weight_variable,
                    input_shape=(maxlen, n_in)))
model.add(Dense(n_out, kernel_initializer=weight_variable))
model.add(Activation('linear'))

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(loss='mean_squared_error',
              optimizer=optimizer)

'''
モデル学習
'''
epochs = 500
batch_size = 10

model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_validation, Y_validation),
          callbacks=[early_stopping])

'''
出力を用いて予測
'''
truncate = maxlen
Z = X[:1]  # 元データの最初の一部だけ切り出し

original = [f[i] for i in range(maxlen)]
predicted = [None for i in range(maxlen)]

for i in range(length_of_sequences - maxlen + 1):
    z_ = Z[-1:]
    y_ = model.predict(z_)
    sequence_ = np.concatenate(
        (z_.reshape(maxlen, n_in)[1:], y_),
        axis=0).reshape(1, maxlen, n_in)
    Z = np.append(Z, sequence_, axis=0)
    predicted.append(y_.reshape(-1))

f_predict = open('result.txt', 'w')
for i in predicted:
    i = str(i)
    f_predict.write(i)
    f_predict.write('\n')
f_predict.close()
