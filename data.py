

import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.image as mpig
from keras import datasets, layers, models, Sequential  # 这里keras版本是2.8.0
from keras.models import Sequential
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from keras.layers import Dense, Dropout

# 读取数据集 验证集、测试集、训练集

# # open and read lables按照labels中的标签分三个数据集
from sklearn.decomposition import PCA

f=open("images/labels.csv")
labels=pd.read_csv(f)
str1=["train","test","dev"]
# for i in (str1):
#     if not os.path.exists('images/'+i):
#         os.mkdir('images/'+i)
# for i in str1:
#     list = labels[labels["set"] == i]
#     l = list["image"].tolist()
#     for each in l:
        # shutil.move('images/images_256/'+each,'images/'+i)
print("start-1")

# read train dataset
train=[]
for filename in os.listdir("images/train"):
    img = mpig.imread("images/train/"+filename)
    img = img.reshape(1, 3 * 256 * 256) / 255.0
    train.append(img)
    # print(img.shape) #(256,256,3)->(1,196608)
f = open("images/train1.csv")
list=pd.read_csv(f)
gender0 = list["gender"]
status0 = list["status"]
train = np.array(train, dtype=np.int64)
status0 = np.array(status0, dtype=np.int64)
gender0 = np.array(gender0, dtype=np.int64)
# 1. 创建文件对象

# test=pd.DataFrame(train)
# test.to_csv('images/data/train.csv')
print(gender0[0],"train已完成")


# read test dataset
test=[]
for filename in os.listdir("images/test"):
    img = mpig.imread("images/test/"+filename)
    img = img.reshape(1, 3 * 256 * 256) / 255.0
    test.append(img)
    # print(img.shape) #(256,256,3)->(1,196608)
list = labels[labels["set"] == "test"]
gender1 = list["gender"]
status1 = list["status"]
gender1 = np.array(gender1, dtype=np.int64)
status1 = np.array(status1, dtype=np.int64)
print(gender1[0],"test已完成")
test = np.array(test, dtype=np.int64)
new=pd.DataFrame(list)
new.to_csv('images/test.csv')


# read dev dataset
dev=[]
for filename in os.listdir("images/dev"):
    img = mpig.imread("images/dev/"+filename)
    img = img.reshape(1, 3 * 256 * 256) / 255.0
    dev.append(img)
    # print(img.shape) #(256,256,3)->(1,196608)
list = labels[labels["set"] == "dev"]
gender2 = list["gender"]
status2 = list["status"]
status2 = np.array(status2, dtype=np.int64)
gender2 = np.array(gender2, dtype=np.int64)
dev = np.array(dev, dtype=np.int64)
print(gender2[0],"devok")

# print(train.shape)
# train = PCA(4238,1,256*3).fit(train)
# train = train.components_


train_images=train
train_labels=gender0
test_images=test
test_labels=gender1
dev_images=dev
dev_labels=gender2




print(train.shape)
# print("test_images",test_images.shape)
# print("test_labels",test_labels.shape)
print("train_labels",train_labels.shape)
print("train_images",train_images.shape)
# print("dev_images",dev_images.shape)
# print("dev_labels",dev_labels.shape)
# train_labels = train_labels.reshape((4238, 256, 256, 3))
# test_labels = test_labels.reshape((527))