"""
猫狗分类 数据集CNN分类
"""
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt

model = Sequential()
model.add(Conv2D(input_shape=(150, 150, 3), filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool2D(2, 2))

model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool2D(2, 2))

model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool2D(2, 2))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
# 绘制网络结构图
plot_model(model, to_file='model_catvsdog.png', show_shapes=True, show_layer_names=True, rankdir='TB')
#数据预处理
train_datagen = ImageDataGenerator(
    rotation_range=40,  # 随机旋转度数
    width_shift_range=0.2,  # 随机水平平移
    height_shift_range=0.2,  # 随机竖直平移
    rescale=1 / 255,  # 数据归一化
    shear_range=0.2,  # 随机错切变换
    zoom_range=0.2,  # 随机放大
    horizontal_flip=True,  # 水平翻转
    fill_mode='nearest'  # 填充方式
)
test_datagen = ImageDataGenerator(
    rescale=1 / 255
)
batch_size = 32
train_generator = train_datagen.flow_from_directory(
    './image/train',
    target_size=(150, 150),
    batch_size=batch_size
)
test_generator = test_datagen.flow_from_directory(
    './image/test',
    target_size=(150, 150),
    batch_size=batch_size
)
# 打印标签字典
print(train_generator.class_indices)
adam = Adam(lr=1e-4)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_generator, steps_per_epoch=len(train_generator), \
                    epochs=10, validation_data=test_generator, \
                    validation_steps=len(test_generator))
model.save('./model/model_catvsdog.h5')

# 测试
# 定义标签
label = np.array(['cat', 'dog'])
model = load_model('./model/model_catvsdog.h5')
image = load_img('./image/test/cat/cat.1002.jpg')
plt.figure(figsize=(10,10))
img = plt.imread('./image/test/cat/cat.1002.jpg')
plt.imshow(img)
plt.axis('off')
plt.show()

image = image.resize((150, 150))
image = img_to_array(image)
image = image / 255
image = np.expand_dims(image, 0)
print(image.shape)
print(label[model.predict_classes(image)])
