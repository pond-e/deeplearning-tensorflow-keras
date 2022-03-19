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


f = [k[1:] for k in l[1:138]]
#print(f)
x = []
for i in range(len(f)):
    y = float(f[i][0])
    x.append(y)
f = np.array(x)
#print(f)

np.random.seed(0)

'''
データの生成
'''
I = 1
noise = np.random.uniform(low=-30, high=30, size=len(f))
l = f+noise
print(l)