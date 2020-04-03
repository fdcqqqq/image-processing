"""
mnist数据集分类
"""
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
#下载不了数据集可以采用以下方式加载
# path = './mnist.npz'
# f = np.load(path)
# x_train, y_train = f['x_train'], f['y_train']
# x_test, y_test = f['x_test'], f['y_test']
# f.close()

(x_train,y_train),(x_test,y_test) = mnist.load_data()
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)
#                reshape  ---->60000,   28*28=784
x_trian = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0
#x_train = x_train.astype('float32')  小数更利于神经网络训练
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# model = load_model('./model/model.h5')#加载模型
model = Sequential([
    Dense(units=200, input_dim=784, bias_initializer='one', activation='tanh', kernel_regularizer=l2(0.0003)),
    Dropout(0.5),
    Dense(units=100, bias_initializer='one', activation='tanh', kernel_regularizer=l2(0.0003)),
    Dropout(0.5),
    Dense(units=10, bias_initializer='one', activation='softmax', kernel_regularizer=l2(0.0003))
])
sgd = SGD(lr=0.2)
#
# model.compile(optimizer=sgd,
#               loss='mse',
#               metrics=['accuracy'])

# 交叉熵函数
model.compile(optimizer=Adam(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_trian, y_train, batch_size=32, epochs=10)
loss, accuracy = model.evaluate(x_test, y_test)
print('\ntest loss', loss)
print('test accuracy:', accuracy)
model.save('./model/model.h5')#保存模型
