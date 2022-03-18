import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import SimpleRNN
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler


import csv

with open('./press_log_100Hz/press_logs_20220309-103302_2_checked.csv') as f:
    reader = csv.reader(f)
    l = [row for row in reader]

f = [k[1:] for k in l[1:]]
x = []
for i in range(len(f)):
    y = float(f[i][0])
    x.append(y)
f = np.array(x)
print(f)

np.random.seed(0)

'''
データの生成
'''
I = 1
noise = np.random.uniform(low=-30, high=30, size=len(f))
l = f+noise

length_of_sequences = len(l)-1
#print(length_of_sequences)
maxlen = 70  # ひとつの時系列データの長さ

data = []
target = []

for i in range(0, length_of_sequences - maxlen + 1):
    data.append(l[i: i + maxlen])
    target.append(l[i + maxlen])

#print(data)

X = np.array(data).reshape(len(data), maxlen, I)
Y = np.array(target).reshape(len(data), I)

# データ設定
N_train = int(len(data) * 0.9)
N_validation = len(data) - N_train

X_train, X_validation, Y_train, Y_validation = \
    train_test_split(X, Y, test_size=N_validation)

# 正規化
sclr_x = MinMaxScaler(feature_range=(0, 1))
sclr_y = MinMaxScaler(feature_range=(0, 1))

X_train = sclr_x.fit_transform(X_train.reshape(-1, 1))
X_validation = sclr_x.transform(X_validation.reshape(-1, 1))

Y_train = sclr_y.fit_transform(Y_train.reshape(-1, 1))
Y_validation = sclr_y.transform(Y_validation.reshape(-1, 1))

'''
モデル設定
'''
n_in = len(X[0][0])  # 入力層次元:1
print(n_in)
n_hidden = 20
n_out = len(Y[0])  # 出力層次元:1