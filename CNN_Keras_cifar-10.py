# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 12:37:24 2018
@author: leodflag
"""
from keras.datasets import cifar10
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)

np.random.seed(10)

#  載入cifar10影像資料
(x_img_train, y_label_train), (x_img_test, y_label_test) = cifar10.load_data()
print('train data:', 'images:', x_img_train.shape,'labels:', y_label_train.shape)
# x_img_train (50000, 32, 32,  3)      y_label_train  (50000, 1)
print('test data:', 'images:', x_img_test.shape, 'labels:', y_label_test.shape)
# x_img_test  (10000, 32, 32, 3)      y_label_test   (10000, 1)

#  圖片特徵值標準化
x_img_train_normalize = x_img_train.astype('float32')/255.0
x_img_test_normalize = x_img_test.astype('float32')/255.0

#  圖片真實值以onehot encoding轉換
y_label_train_OneHot = np_utils.to_categorical(y_label_train)
y_label_test_OneHot = np_utils.to_categorical(y_label_test)

#  建立線性堆疊模型
model = Sequential()
#  卷積層 1  import 32*32 *3  output 32*32*32
model.add(Conv2D(filters=32, kernel_size=(3, 3),
                 input_shape=(32, 32, 3),
                 activation='relu',
                 padding='same'))
#  加入Dropout ，每次訓練迭代時，會隨機在神經網路中放棄25%的神經元，避免overfitting
model.add(Dropout(rate=0.25))
#  建立池化層1
model.add(MaxPooling2D(pool_size=(2, 2)))  # 16*16 32
#  卷積層 2
model.add(Conv2D(filters=64, kernel_size=(3, 3),  # 16*16  64
                 activation='relu', padding='same'))
#  dropout
model.add(Dropout(0.25))
#  建立池化層 2
model.add(MaxPooling2D(pool_size=(2, 2)))  # 8*8 64
#  建立平坦層  8*8 64  = 4096
model.add(Flatten())
model.add(Dropout(rate=0.25))
#  建立隱藏層
model.add(Dense(1024, activation='relu'))
model.add(Dropout(rate=0.25))
#  建立輸出層
model.add(Dense(10, activation='softmax'))
#  查看模型摘要
print(model.summary())

#  定義訓練方法
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

#  開始訓練  0.2為驗證資料  batch_size每次訓練128筆資料  verbose=2時會顯示訓練過程
model.fit(x_img_train_normalize, y_label_train_OneHot,
                          validation_split=0.2,
                          epochs=10, batch_size=128, verbose=1)
#  衡量模型訓練誤差
score = model.evaluate(x_img_test_normalize, y_label_test_OneHot,
                       verbose=0)  # 同fit的verbose，預設1，只能取0、1
print('Test loss:', score[0])  # 0.0514164837168064
print('Test accuracy:', score[1])  # 0.983299970626831

