"""
狗分类 数据集VGG16分类
训练部分
"""
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model
import json

batch_size = 32
train_data = './data/train/'
test_data = './data/test/'
img_h = 150
img_w = 150

# include_top=False 不包含全连接层
vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_h, img_w, 3))
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(10, activation='softmax'))

model = Sequential()
model.add(vgg16_model)
model.add(top_model)
# model.summary()
# 绘制网络结构图
plot_model(model, to_file='model_vgg_dog.png', show_shapes=True, show_layer_names=True, rankdir='TB')
train_datagen = ImageDataGenerator(
    rotation_range=40,  # 随机旋转度数
    width_shift_range=0.2,  # 随机水平平移
    height_shift_range=0.2,  # 随机竖直平移
    rescale=1 / 255,  # 数据归一化
    shear_range=20,  # 随机错切变换
    zoom_range=0.2,  # 随机放大
    horizontal_flip=True,  # 水平翻转
    fill_mode='nearest'  # 填充方式
)
test_datagen = ImageDataGenerator(
    rescale=1 / 255
)
batch_size = 32
train_generator = train_datagen.flow_from_directory(
    train_data,
    target_size=(150, 150),
    batch_size=batch_size
)
test_generator = test_datagen.flow_from_directory(
    test_data,
    target_size=(150, 150),
    batch_size=batch_size
)
# 打印标签字典
# print(train_generator.class_indices)
label = train_generator.class_indices
#把标签的键值对调换下
label = dict(zip(label.values(), label.keys()))
#写入json文件
file = open('label.json', 'w', encoding='utf-8')
json.dump(label, file)
print(label)

sgd = SGD(lr=1e-4, momentum=0.9)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_generator, steps_per_epoch=len(train_generator), \
                    epochs=1, validation_data=test_generator, \
                    validation_steps=len(test_generator))

model.save('./model/model_vgg16_dog.h5')
