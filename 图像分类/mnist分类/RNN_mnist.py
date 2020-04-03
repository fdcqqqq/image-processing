"""
mnist数据集RNN分类
"""
import numpy as np
from keras.models import Sequential
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPool2D, Flatten
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD, Adam
from keras.regularizers import l2

# 数据长度一行有28个像素
input_size = 28
# 序列长度一共28行
time_steps = 28
cell_size = 50

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

x_trian = x_train / 255.0
x_test = x_test / 255.0

y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# model = load_model('./model/model_cnn.h5')#加载模型
model = Sequential()
model.add(SimpleRNN(
    units=cell_size,  # 输出
    input_shape=(time_steps, input_size)  # 输入
))

model.add(Dense(10, activation='softmax'))
adam = Adam(lr=1e-4)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_trian, y_train, batch_size=64, epochs=10)
loss, accuracy = model.evaluate(x_test, y_test)

print('test losss:', loss)
print('test accuracy:', accuracy)
model.save('./model/model_cnn.h5')  # 保存模型
