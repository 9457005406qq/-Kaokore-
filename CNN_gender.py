import keras
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D
from keras.layers import Dense, Flatten
import csv
from data import test_images,test_labels,train_labels,train_images,dev_images,dev_labels,labels
import pandas as pd
train_labels= np.array(train_labels)
test_labels= np.array(test_labels)
dev_labels= np.array(dev_labels)

from matplotlib import pyplot as plt
fig1=plt.figure(figsize=(8,5))#建立图形
plt.plot(test_labels)#画图
# 加标注
plt.title('Test:Gender information')
plt.xlabel('Photo Num')
plt.ylabel('Gender')
plt.show()

#构建模型CNN
test_images=test_images.reshape(527,256,256,3)
train_images=train_images.reshape(4238,256,256,3)
dev_images=dev_images.reshape(533,256,256,3)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), 
                 activation='relu', input_shape=(256, 256,3)))
model.add(MaxPool2D(pool_size=(2,2), strides=(5,5)))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(4,4), strides=(3,3)))
model.add(Flatten()) 
model.add(Dense(1000, activation='relu'))
model.add(Dense(2, activation='softmax'))
#模型编译
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#训练
model.fit(train_images, train_labels, batch_size=128, 
          epochs=10,validation_data=(dev_images,dev_labels))

#评估模型
score = model.evaluate(test_images, test_labels)
print('acc', score[1])
print('loss',score[0])


CNN_g=[]
pre = model.predict(test_images)  # 对所有测试图片进行预测
for x in range(len(test_images)):
    a=np.array(pre[x]).argmax()
    print(a)
    CNN_g.append(a)


from matplotlib import pyplot as plt

fig3=plt.figure(figsize=(12,6))#建立图形
plt.plot(test_labels,label='real test')#画图
plt.plot(CNN_g,label='predict test')#画图
#加标注
plt.xticks(rotation=-90) 
plt.title('CNN gender')
plt.xlabel('time')
plt.ylabel('comprise')
plt.legend()
plt.show()



# 存储数据
result = pd.DataFrame(CNN_g,columns=['predict_test'])
result.to_csv('CNN_predict_gender.csv')