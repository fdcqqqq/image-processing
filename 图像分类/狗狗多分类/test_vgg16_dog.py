"""
狗分类 数据集VGG16分类
测试部分
"""
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
import json

file = open('label.json', 'r', encoding='utf-8')
label = json.load(file)

model = load_model('./model/model_vgg16_dog.h5')


def predict(image):
    image = load_img(image)
    plt.imshow(image)
    image = image.resize((150, 150))
    image = img_to_array(image)
    image = image / 255
    image = np.expand_dims(image, 0)
    plt.axis('off')
    plt.title(label[str(model.predict_classes(image)[0])])
    plt.show()


predict('./data/test/n02086240-ShihTzu/Chrysanthemum-0c179a58922bf966.jpg')
