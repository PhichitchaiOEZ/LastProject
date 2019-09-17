import pandas as pd
import csv
import numpy as np
from scipy.stats import iqr,skew  # use IQR,Skewness
import matplotlib.pyplot as plt

Ax = []
Ay = []
Az = []
Gx = []
Gy = []
Gz = []
# Ax = [] , Ay = [] , Az = [] , Gx = [] , Gy = [] , Gz = []
maxAx = []
maxAy = []
maxAz = []
maxGx = []
maxGy = []
maxGz = []
minAx = []
minAy = []
minAz = []
minGx = []
minGy = []
minGz = []
meanAx = []
meanAy = []
meanAz = []
meanGx = []
meanGy = []
meanGz = []
varianceAx = []
varianceAy = []
varianceAz = []
varianceGx = []
varianceGy = []
varianceGz = []
stdAx = []
stdAy = []
stdAz = []
stdGx = []
stdGy = []
stdGz = []
medianAx = []
medianAy = []
medianAz = []
medianGx = []
medianGy = []
medianGz = []
rmsAx = []
rmsAy = []
rmsAz = []
rmsGx = []
rmsGy = []
rmsGz = []

iqrAx = []
iqrAy = []
iqrAz = []
iqrGx = []
iqrGy = []
iqrGz = []

iominAx = []
iominAy = []
iominAz = []
iominGx = []
iominGy = []
iominGz = []

madAx = []
madAy = []
madAz = []
madGx = []
madGy = []
madGz = []

skAx = []
skAy = []
skAz = []
skGx = []
skGy = []
skGz = []


#data = open('/content/drive/My Drive/Project/Colab/sitOH.csv')
data = open('D:\Last_Project\Dataset\checkOHLR.csv')
data_r = csv.reader(data, delimiter=',')
for row in data_r:
    c1, c2, c3, c4, c5, c6 = row
    Ax.append(c1)
    Ay.append(c2)
    Az.append(c3)
    Gx.append(c4)
    Gy.append(c5)
    Gz.append(c6)
data.close()

Ax = [float(x) for x in Ax]
Ay = [float(x) for x in Ay]
Az = [float(x) for x in Az]
Gx = [float(x) for x in Gx]
Gy = [float(x) for x in Gy]
Gz = [float(x) for x in Gz]


def feature():
    for i in range(0, len(Ax), 7):
        ax = Ax[i:i + 14]
        ay = Ay[i:i + 14]
        az = Az[i:i + 14]
        gx = Gx[i:i + 14]
        gy = Gy[i:i + 14]
        gz = Gz[i:i + 14]
        # -------MaxValue--------#
        maxAx.append(np.amax(ax))
        maxAy.append(np.amax(ay))
        maxAz.append(np.amax(az))

        maxGx.append(np.amax(gx))
        maxGy.append(np.amax(gy))
        maxGz.append(np.amax(gz))
        # -------MinValue--------#
        minAx.append(np.amin(ax))
        minAy.append(np.amin(ay))
        minAz.append(np.amin(az))

        minGx.append(np.amin(gx))
        minGy.append(np.amin(gy))
        minGz.append(np.amin(gz))
        # -------MeanValue--------#
        meanAx.append(np.mean(ax))
        meanAy.append(np.mean(ay))
        meanAz.append(np.mean(az))

        meanGx.append(np.mean(gx))
        meanGy.append(np.mean(gy))
        meanGz.append(np.mean(gz))
        # -------MedianValue--------#
        medianAx.append(np.median(ax))
        medianAy.append(np.median(ay))
        medianAz.append(np.median(az))

        medianGx.append(np.median(gx))
        medianGy.append(np.median(gy))
        medianGz.append(np.median(gz))
        # ---StandardDeviationValue---#
        stdAx.append(np.std(ax))
        stdAy.append(np.std(ay))
        stdAz.append(np.std(az))

        stdGx.append(np.std(gx))
        stdGy.append(np.std(gy))
        stdGz.append(np.std(gz))
        # -------VarianValue--------#
        varianceAx.append(np.var(ax))
        varianceAy.append(np.var(ay))
        varianceAz.append(np.var(az))

        varianceGx.append(np.var(gx))
        varianceGy.append(np.var(gy))
        varianceGz.append(np.var(gz))
        # -------RootMeanSquareValue--------#
        rmsAx.append(np.sqrt(np.mean(power(ax))))
        rmsAy.append(np.sqrt(np.mean(power(ay))))
        rmsAz.append(np.sqrt(np.mean(power(az))))

        rmsGx.append(np.sqrt(np.mean(power(gx))))
        rmsGy.append(np.sqrt(np.mean(power(gy))))
        rmsGz.append(np.sqrt(np.mean(power(gz))))

        # -------IQR--------#
        iqrAx.append(iqr(ax, rng=(25, 75), interpolation='midpoint'))
        iqrAy.append(iqr(ay, rng=(25, 75), interpolation='midpoint'))
        iqrAz.append(iqr(az, rng=(25, 75), interpolation='midpoint'))

        iqrGx.append(iqr(gx, rng=(25, 75), interpolation='midpoint'))
        iqrGy.append(iqr(gy, rng=(25, 75), interpolation='midpoint'))
        iqrGz.append(iqr(gz, rng=(25, 75), interpolation='midpoint'))

        # -------IndexOfMinimum Value--------#
        iominAx.append(np.argmin(ax))
        iominAy.append(np.argmin(ay))
        iominAz.append(np.argmin(az))

        iominGx.append(np.argmin(gx))
        iominGy.append(np.argmin(gy))
        iominGz.append(np.argmin(gz))

        # -------Mean Absolute Deviation--------#
        madAx.append(pd.Series(ax).mad())
        madAy.append(pd.Series(ay).mad())
        madAz.append(pd.Series(az).mad())
        madGx.append(pd.Series(gx).mad())
        madGy.append(pd.Series(gy).mad())
        madGz.append(pd.Series(gz).mad())

        # -------Skewness--------#
        skAx.append(skew(ax))
        skAy.append(skew(ay))
        skAz.append(skew(az))
        skGx.append(skew(gx))
        skGy.append(skew(gy))
        skGz.append(skew(gz))

def power(listpow):
    return [x ** 2 for x in listpow]
feature()
print(meanAx[0])
print(medianAx[0])
print(stdAx[0])
print(skAx[0])
x = 3*(meanAx[0]-medianAx[0])/stdAx[0]
print(x)
# Plot with differently-colored markers.
t = []
for i in range(len(maxAx)):
    t.append(i)
plt.plot(t, rmsAx, 'b-', label='X')
plt.plot(t, rmsAy, 'g-', label='Y')
plt.plot(t, rmsAz, 'r-', label='Z')

# Create legend.
plt.legend(loc='upper left')
plt.xlabel('value')
plt.ylabel('data')
plt.show()

df = pd.DataFrame(list(zip(maxAx, maxAy,maxAz,maxGx,maxGy,maxGz,minAx,minAy,minAz,minGx,minGy,minGz,meanAx,meanAy,meanAz,meanGx,meanGy,meanGz,
                           medianAx,medianAy,medianAz,medianGx,medianGy,medianGz,stdAx,stdAy,stdAz,stdGx,stdGy,stdGz,varianceAx,varianceAy,varianceAz,varianceGx,varianceGy,varianceGz,
                           rmsAx,rmsAy,rmsAz,rmsGx,rmsGy,rmsGz,iqrAx,iqrAy,iqrAz,iqrGx,iqrGy,iqrGz,iominAx,iominAy,iominAz,iominGx,iominGy,iominGz,madAx,madAy,madAz,madGx,madGy,madGz)),
               columns =['maxAx', 'maxAy','maxAz','maxGx','maxGy','maxGz','minAx','minAy','minAz','minGx','minGy','minGz','meanAx','meanAy','meanAz','meanGx','meanGy','meanGz',
                         'medianAx','medianAy','medianAz','medianGx','medianGy','medianGz','stdAx','stdAy','stdAz','stdGx','stdGy','stdGz','varianceAx','varianceAy','varianceAz','varianceGx','varianceGy','varianceGz',
                         'rmsAx','rmsAy','rmsAz','rmsGx','rmsGy','rmsGz','iqrAx','iqrAy','iqrAz','iqrGx','iqrGy','iqrGz','iominAx','iominAy','iominAz','iominGx','iominGy','iominGz','madAx','madAy','madAz','madGx','madGy','madGz'])
export_csv = df.to_csv (r'D:\Last_Project\Dataset\Feature_checkLR.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path
print(df)