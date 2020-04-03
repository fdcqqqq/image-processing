"""
mnist数据集CNN分类
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPool2D, Flatten
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt

(x_train,y_train),(x_test,y_test) = mnist.load_data()
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)

x_trian = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)
# model = load_model('./model/model_cnn.h5')#加载模型
model = Sequential()
model.add(Conv2D(
    input_shape=(28, 28, 1),
    filters=32,
    kernel_size=5,
    strides=1,
    padding='same',
    activation='relu'
))
model.add(MaxPool2D(
    pool_size=2,
    strides=2,
    padding='same'
))

model.add(Conv2D(64, 5, strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(2, 2, 'same'))
# 扁平化输出为1维
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

adam = Adam(lr=1e-4)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_trian, y_train, batch_size=64, epochs=2)
loss, accuracy = model.evaluate(x_test, y_test)

print('test losss:', loss)
print('test accuracy:', accuracy)
model.save('./model/model_cnn.h5')#保存模型

#绘制网络结构图
plot_model(model, to_file='model_cnn.png', show_shapes=True,show_layer_names=True,rankdir='TB')
plt.figure(figsize=(10,10))
img = plt.imread('model_cnn.png')
plt.imshow(img)
plt.axis('off')
plt.show()