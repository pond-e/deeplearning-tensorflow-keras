from re import S
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler


import csv

with open('./press_log_100Hz/press_logs_20220309-103302_2_checked.csv') as f:
    reader = csv.reader(f)
    l = [row for row in reader]

end__ = 138

f = [k[1:] for k in l[1:end__]]
for i in range(len(f)):
    for j in range(len(f[0])):
        f[i][j] = float(f[i][j])

#print(len(f))
#print(len(f[0]))

with open('./press_log_100Hz/press_logs_20220303-140302_0.csv') as f_2:
    reader = csv.reader(f_2)
    l_2 = [row for row in reader]

f_2 = [k[1:] for k in l_2[1:end__]]
for i in range(len(f_2)):
    for j in range(len(f_2[0])):
        f_2[i][j] = float(f_2[i][j])
#print(len(f_2))
#print(len(f_2[0]))

read_file = []
read_file.append(f)
read_file.append(f_2)
#print(len(read_file))
Y = np.array([[0, 0, 1], [1, 0, 1]])
# or
Y = np.array([[0,1], [1, 0]])

read_file = np.array(read_file)


np.random.seed(0)

'''
データの生成
'''
sample_number = 2
I = 3
#noise = np.random.uniform(low=-30, high=30, size=(len(f), I))
#l = f+noise
l = read_file

# 正規化
sclr_x = MinMaxScaler(feature_range=(0, 1))
# TODO:サンプル,データ,次元に変更する
l[0] = sclr_x.fit_transform(l[0].reshape(-1, I))
l[1] = sclr_x.fit_transform(l[1].reshape(-1, I))

length_of_sequences = len(l[0])-1
maxlen = 50  # ひとつの時系列データの長さ

data_tmp = []
data = []

for i in range(0, length_of_sequences - maxlen +1):
    data_tmp.append(l[0][i: i + maxlen])
data_tmp = np.array(data_tmp)
data.append(data_tmp)

data_tmp = []
for i in range(0, length_of_sequences - maxlen +1):
    data_tmp.append(l[1][i: i + maxlen])
data_tmp = np.array(data_tmp)
data.append(data_tmp)

data = np.array(data)
X = data
print(len(X))
#Y = np.array(target).reshape(len(data), I)

# データ設定
N_train = int(len(data) * 0.9)
N_validation = len(data) - N_train
X_train_1, X_validation_1, Y_train_1, Y_validation_1 = train_test_split(X, Y, test_size=N_validation)
print(Y_train_1)



