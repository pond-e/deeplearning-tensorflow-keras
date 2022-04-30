#!/usr/bin/env python3

import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
import numpy as np
import random

input_dim = 1                # 入力データの次元数：実数値1個なので1を指定
output_dim = 1               # 出力データの次元数：同上
num_hidden_units = 128       # 隠れ層のユニット数
len_sequence = 10            # 時系列の長さ
batch_size = 300             # ミニバッチサイズ
num_of_training_epochs = 100 # 学習エポック数
learning_rate = 0.001        # 学習率
num_training_samples = 1000  # 学習データのサンプル数

# データを作成
def create_data(nb_of_samples, sequence_len):
    # 乱数で {0.0, 1.0} の列を指定された個数だけ生成する
    X = np.random.randint(0, 2, (nb_of_samples, sequence_len)).astype("float32")
    # 各行の総和を正解ラベルとする
    t = np.sum(X, axis=1)
    # LSTMに与える入力は (サンプル, 時刻, 特徴量の次元) の3次元になる。
    return X.reshape((nb_of_samples, sequence_len, 1)), t

# 乱数シードを固定値で初期化
random.seed(0)
np.random.seed(0)
tf.set_random_seed(0)

X, t = create_data(num_training_samples, len_sequence)

# モデル構築
model = Sequential()
model.add(LSTM(
    num_hidden_units,
    input_shape=(len_sequence, input_dim),
    return_sequences=False))
model.add(Dense(output_dim))
model.compile(loss="mean_squared_error", optimizer=Adam(lr=learning_rate))
model.summary()

print(X)
print(t)
# 学習
model.fit(
    X, t,
    batch_size=batch_size,
    epochs=num_of_training_epochs,
    validation_split=0.1
)

# 予測
# (サンプル, 時刻, 特徴量の次元) の3次元の入力を与える。
test = np.array([1, 1, 1, 1, 0, 1, 1, 1, 0, 1]).astype("float32").reshape((1, 10, 1))
print(model.predict(test)) # [[7.7854743]]