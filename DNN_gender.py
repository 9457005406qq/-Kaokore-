import cv2
import os
import numpy as np
import pandas as pd
from data import test_images,test_labels,train_labels,train_images,dev_images,dev_labels
from keras.models import Sequential
from keras.layers import Dense, Dropout




print("start DNN  gender")
train_images=train_images.reshape((4238, 3* 256*256 ))
test_images = test_images.reshape((527, 256*3*256))
dev_images=dev_images.reshape((533,256*3*256))



from matplotlib import pyplot as plt
fig1=plt.figure(figsize=(8,5))#建立图形
plt.plot(test_labels)#画图
# 加标注
plt.title('Test:Gender information')
plt.xlabel('Photo Num')
plt.ylabel('Gender')
plt.show()
# 构建 DNN 网络模型
model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(3*256*256,)))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
# model.save_weights('./checkpoints/my_checkpoint')
model.add(Dense(2, activation='softmax'))
DNN_scores=[]
# Compile the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])


model.fit(train_images, train_labels, batch_size=128, epochs=5,
          validation_data=(dev_images, dev_labels), verbose=1)
# model.fit(train_images, train_labels,epochs=1,
#           batch_size=32,shuffle=True,verbose=2,validation_data=(dev_images, dev_labels))
# model.save_weights('./checkpoints/my_checkpoint')

# Evaluate the model on test data
DNN_score = model.evaluate(test_images, test_labels, verbose=0)
DNN_scores.append(DNN_score)

print(f"Test loss: {DNN_score[0]}")
print(f"Test accuracy: {DNN_score[1]}")
DNN_g=[]
pre = model.predict(test_images)  # 对所有测试图片进行预测
for x in range(len(test_images)):
    a=np.array(pre[x]).argmax()
    print(a)
    DNN_g.append(a)
    # print(test_labels[x])

from matplotlib import pyplot as plt

fig3=plt.figure(figsize=(12,6))#建立图形
plt.plot(test_labels,label='real test')#画图
plt.plot(DNN_g,label='predict test')#画图
#加标注
plt.xticks(rotation=-90) 
plt.title('DNN gender')
plt.xlabel('time')
plt.ylabel('comprise')
plt.legend()
plt.show()



# 存储数据
result = pd.DataFrame(DNN_g,columns=['predict_test'])
result.to_csv('DNN_predict_gender.csv')


    
    
    