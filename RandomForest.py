import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# Feature Scaling
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

names = ['maxAx', 'maxAy','maxAz','maxGx','maxGy','maxGz','minAx','minAy','minAz','minGx','minGy','minGz','meanAx','meanAy','meanAz','meanGx','meanGy','meanGz',
                          'medianAx','medianAy','medianAz','medianGx','medianGy','medianGz','stdAx','stdAy','stdAz','stdGx','stdGy','stdGz','varianceAx','varianceAy','varianceAz','varianceGx','varianceGy','varianceGz',
                          'rmsAx','rmsAy','rmsAz','rmsGx','rmsGy','rmsGz','iqrAx','iqrAy','iqrAz','iqrGx','iqrGy','iqrGz','iominAx','iominAy','iominAz','iominGx','iominGy','iominGz','madAx','madAy','madAz','madGx','madGy','madGz', 'Class']


dataset = pd.read_csv('D:\Last_Project\Dataset\DataTESTusenum.csv',names = names)
print(dataset.head())

X = dataset.iloc[:, 0:60].values
y = dataset.iloc[:, 60].values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

regressor = RandomForestRegressor(n_estimators=20, random_state=40)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


errors = abs(y_pred - y_test)
# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))
# print(accuracy_score(y_test, y_pred))

# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
# print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

