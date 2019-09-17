import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.externals import joblib

url = 'D:\Last_Project\Dataset\DataTEST.csv'
url2 = 'D:\Last_Project\Dataset\DataTestModel.csv'
# Assign colum names to the dataset
names = ['maxAx', 'maxAy','maxAz','maxGx','maxGy','maxGz','minAx','minAy','minAz','minGx','minGy','minGz','meanAx','meanAy','meanAz','meanGx','meanGy','meanGz',
                          'medianAx','medianAy','medianAz','medianGx','medianGy','medianGz','stdAx','stdAy','stdAz','stdGx','stdGy','stdGz','varianceAx','varianceAy','varianceAz','varianceGx','varianceGy','varianceGz',
                          'rmsAx','rmsAy','rmsAz','rmsGx','rmsGy','rmsGz','iqrAx','iqrAy','iqrAz','iqrGx','iqrGy','iqrGz','iominAx','iominAy','iominAz','iominGx','iominGy','iominGz','madAx','madAy','madAz','madGx','madGy','madGz', 'Class']

# Read dataset to pandas dataframe
dataset = pd.read_csv(url, names=names)
datatest = pd.read_csv(url2, names=names)
print(dataset.head())

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 60].values

Xtest = datatest.iloc[:, :-1].values
ytest = datatest.iloc[:, 60].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# from sklearn.preprocessing import StandardScaler #ทำการ Standardizing ข้อมูลให้อยู่ในสเกลเดียวกันเพื่อประสิทธิภาพในการเทรนโมเดล
# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)
#
# Xtest = scaler.transform(Xtest)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)


# save the model to disk
# filename = 'EBM_model.ooo'
# joblib.dump(classifier,filename)


y_pred = classifier.predict(X_test)
# pre2 = classifier.predict(Xtest)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


print("-------------------------------------------------------------------------------")
# print(confusion_matrix(ytest, pre2))