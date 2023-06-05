import pandas as pd
import numpy as np

test=pd.read_csv('images/test.csv')
DNN_status=pd.read_csv('DNN_predict_status.csv')
DNN_gender=pd.read_csv('DNN_predict_gender.csv')
CNN_status=pd.read_csv('CNN_predict_status.csv')
CNN_gender=pd.read_csv('CNN_predict_gender.csv')

CNN_gender=CNN_gender['predict_test']
CNN_status=CNN_status['predict_test']
DNN_gender=DNN_gender['predict_test']
DNN_status=DNN_status['predict_test']
test=test['image']

data=pd.concat([test,CNN_gender,CNN_status],axis=1)
df=pd.DataFrame(data)
df.columns={'image':test,'gender':CNN_gender,'status':CNN_status}
df.to_csv('CNN.csv')
print(df)


data1=pd.concat([test,DNN_gender,DNN_status],axis=1)
df1=pd.DataFrame(data1)
df1.columns={'image':test,'gender':DNN_gender,'status':DNN_status}
df1.to_csv('DNN.csv')
print(df1)
