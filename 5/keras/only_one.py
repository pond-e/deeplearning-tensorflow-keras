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

X = np.array(data).reshape(len(data), maxlen, 1)
Y = np.array(target).reshape(len(data), 1)

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


# データ変換
def convert(X, y, ws = 1, dim = 1):
    data = []
    target = []
    
    #windowサイズが1の場合は全部のデータを使う
    #そうでない場合は、wsで全てのデータが収まる範囲で使う
    if(ws == 1):
        itr = len(X)
    else:
        itr = len(X) - ws
    
    for i in range(itr):
        data.append(X[i: i + ws])
        target.append(y[i])
        
    # データの整形
    data = np.array(data).reshape(len(data), ws, dim)
    target = np.array(target).reshape(-1,1)
        
    return data, target

#変換
X_train, Y_train = convert(X_train, Y_train, maxlen, 1)
X_validation, Y_validation = convert(X_validation, Y_validation, maxlen, 1)

'''
モデル設定
'''
n_in = len(X[0][0])  # 入力層次元:1
n_hidden = 20
n_out = len(Y[0])  # 出力層次元:1


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

hist = model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_validation, Y_validation),
          callbacks=[early_stopping])

'''
出力を用いて予測
'''
truncate = maxlen
Z = X[:1]  # 元データの最初の一部だけ切り出し

#original = [f[i] for i in range(maxlen)]
predicted = [None for i in range(maxlen)]

for i in range(length_of_sequences - maxlen + 1):
    z_ = Z[-1:]
    y_ = model.predict(z_)
    sequence_ = np.concatenate((z_.reshape(maxlen, n_in)[1:], y_),axis=0).reshape(1, maxlen, n_in)
    Z = np.append(Z, sequence_, axis=0)
    predicted.append(y_.reshape(-1))

predicted = np.array(predicted)
pred_inv = sclr_x.inverse_transform(predicted.reshape(-1, 1))


fmt_name = "result.csv"
f_press = open(fmt_name, 'w')   # 書き込みファイル

for i in predicted:
    if(i != None):
        value = "%6.9f"%(i)
        f_press.write(value + "\n")
f_press.close()

'''
学習の進み具合を可視化
'''


#val_acc = hist.history['val_acc']
val_loss = hist.history['val_loss']

plt.rc('font', family='serif')
fig = plt.figure()
plt.plot(range(len(val_loss)), val_loss, label='loss', color='black')
plt.xlabel('epochs')
plt.savefig('only_learn.png')
