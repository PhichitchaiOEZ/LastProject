import numpy as np
import pandas as pd
import sys
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

def predictActivity (data):
    filename = 'EBM_model.ooo'
    # load the model from disk
    model = joblib.load(filename)

    y = model.predict(data)
    return y

print(sys.version)
url2 = 'D:\Last_Project\Dataset\DataTestModel.csv'
datatest = pd.read_csv(url2)
n = 7
d1 =  datatest.iloc[n,:-1].values

print(datatest.iloc[n,60])

d = d1.reshape(-1,1)
print(type(d))
# scaler = StandardScaler()
# scaler.fit(d)
# d = scaler.transform(d)
d2= np.transpose(d)

y =predictActivity (d2)
aa = y[0]
print("------------")
print(aa)
print(type(aa))



