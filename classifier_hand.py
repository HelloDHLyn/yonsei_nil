import numpy
import pandas
from pandas import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')

from dataloader import RawDataLoader, DataLoader

raw_loader = RawDataLoader()

combined_data = []

for group in range(1, 11):
    data_raw = raw_loader.load_eval(str(group))

    for idx in data_raw.index:
        data = data_raw.ix[idx]
        try:
            data_sensor = raw_loader.load_sensor_data(data['File name'])
        except:
            continue

        posture = data[' Input posture']
        if posture == 'Both':
            posture = 1
        elif posture == 'Right':
            posture = 2
        else:
            posture = 0
        
        combined_data.append([posture,
                    data_sensor.acc_x.var(),
                    data_sensor.acc_y.var(),
                    data_sensor.acc_z.var(),
                    data_sensor.azim.var(),
                    data_sensor.roll.var(),
                    data_sensor.gyro_y.var(),
                    data_sensor.gyro_z.var(),
                    data_sensor.acc_x.mean(),
                    data_sensor.acc_y.mean(),
                    data_sensor.acc_z.mean(),
                    data_sensor.azim.mean(),
                    data_sensor.roll.mean(),
                    data_sensor.gyro_y.mean(),
                    data_sensor.gyro_z.mean()])
        
df = DataFrame(combined_data, columns=['posture', 'acc_x', 'acc_y', 'acc_z', 'azim', 'roll', 'gyro_y', 'gyro_z', 'macc_x', 'macc_y', 'macc_z', 'mazim', 'mroll', 'mgyro_y', 'mgyro_z'])
df.to_csv('data/train.csv')

train_data = read_csv('data/train.csv')

trainY = train_data['posture'].values
trainX = train_data.drop('posture', 1)

test_data = []
for group in range(1, 3):
    data_raw = raw_loader.load_eval('test' + str(group))

    for idx in data_raw.index:
        data = data_raw.ix[idx]
        data_sensor = raw_loader.load_sensor_data(data['File name'])

        posture = data[' Input posture']
        if posture == 'Both':
            posture = 1
        elif posture == 'Right':
            posture = 2
        else:
            posture = 0


        test_data.append([posture,
                          data_sensor.acc_x.var(),
                          data_sensor.acc_y.var(),
                          data_sensor.acc_z.var(),
                          data_sensor.azim.var(),
                          data_sensor.roll.var(),
                          data_sensor.gyro_y.var(),
                          data_sensor.gyro_z.var(),
                          data_sensor.acc_x.mean(),
                          data_sensor.acc_y.mean(),
                          data_sensor.acc_z.mean(),
                          data_sensor.azim.mean(),
                          data_sensor.roll.mean(),
                          data_sensor.gyro_y.mean(),
                          data_sensor.gyro_z.mean()])

df = DataFrame(test_data, columns=['posture', 'acc_x', 'acc_y', 'acc_z', 'azim', 'roll', 'gyro_y', 'gyro_z', 'macc_x', 'macc_y', 'macc_z', 'mazim', 'mroll', 'mgyro_y', 'mgyro_z'])
df.to_csv('data/test.csv')

test_data = read_csv('data/test.csv')

testY = test_data['posture'].values
testX = test_data.drop('posture', 1)

rf = RandomForestClassifier(n_estimators=100, max_depth=30, max_leaf_nodes=100)
rf.fit(trainX, trainY)

predTestY = rf.predict(testX)
accuracyTest = float(np.sum(predTestY == testY)) / predTestY.shape[0]

print "testSet accuracy: ", accuracyTest*100, "%"