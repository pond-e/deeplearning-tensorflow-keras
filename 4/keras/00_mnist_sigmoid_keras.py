import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

np.random.seed(0)

'''
データの生成
'''
mnist = fetch_openml('mnist_784', version=1,)

n = len(mnist.data)
print(n)
N = 784  # MNISTの一部を使う
indices = np.random.permutation(range(n))[:N]  # ランダムにN枚を選択

print(mnist.data['pixel1'])
print(len(mnist.data['pixel1']))
print(mnist.data)

X = mnist.data[indices]
y = mnist.target[indices]
Y = np.eye(10)[y.astype(int)]  # 1-of-K 表現に変換

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)

'''
モデル設定
'''
n_in = len(X[0])  # 784
n_hidden = 200
# n_hidden = 4000
n_out = len(Y[0])  # 10

model = Sequential()
model.add(Dense(n_hidden, input_dim=n_in))
model.add(Activation('sigmoid'))

# model.add(Dense(n_hidden))
# model.add(Activation('sigmoid'))

# model.add(Dense(n_hidden))
# model.add(Activation('sigmoid'))

# model.add(Dense(n_hidden))
# model.add(Activation('sigmoid'))

model.add(Dense(n_out))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.01),
              metrics=['accuracy'])

'''
モデル学習
'''
epochs = 100
batch_size = 200

model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)

'''
予測精度の評価
'''
loss_and_metrics = model.evaluate(X_test, Y_test)
print(loss_and_metrics)
