import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


url = 'D:\Last_Project\Dataset\DataTEST.csv'
names = ['maxAx', 'maxAy','maxAz','maxGx','maxGy','maxGz','minAx','minAy','minAz','minGx','minGy','minGz','meanAx','meanAy','meanAz','meanGx','meanGy','meanGz',
                          'medianAx','medianAy','medianAz','medianGx','medianGy','medianGz','stdAx','stdAy','stdAz','stdGx','stdGy','stdGz','varianceAx','varianceAy','varianceAz','varianceGx','varianceGy','varianceGz',
                          'rmsAx','rmsAy','rmsAz','rmsGx','rmsGy','rmsGz','iqrAx','iqrAy','iqrAz','iqrGx','iqrGy','iqrGz','iominAx','iominAy','iominAz','iominGx','iominGy','iominGz','madAx','madAy','madAz','madGx','madGy','madGz', 'Class']


dataset = pd.read_csv(url, names=names)
X = dataset.drop('Class', axis=1)
y = dataset['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
svclassifier = SVC(kernel='poly', degree=8)
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))