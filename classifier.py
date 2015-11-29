import numpy
import pandas
from pandas import *
from sklearn.ensemble import RandomForestClassifier

import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')

from dataloader import RawDataLoader

raw_loader = RawDataLoader()
combined_data = []

for group in range(1, 4):
    data_raw = raw_loader.load_eval(str(group))

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
            
        age = data[' Age']
        
        combined_data.append([posture,
                    age,
                    year,
                    data_sensor.acc_x.var(),
                    data_sensor.acc_y.var(),
                    data_sensor.acc_z.var(),
                    data_sensor.azim.var(),
                    data_sensor.pitch.var(),
                    data_sensor.roll.var(),
                    data_sensor.gyro_x.var(),
                    data_sensor.gyro_y.var(),
                    data_sensor.gyro_z.var(),
                    data_sensor.acc_x.mean(),
                    data_sensor.acc_y.mean(),
                    data_sensor.acc_z.mean(),
                    data_sensor.azim.mean(),
                    data_sensor.pitch.mean(),
                    data_sensor.roll.mean(),
                    data_sensor.gyro_x.mean(),
                    data_sensor.gyro_y.mean(),
                    data_sensor.gyro_z.mean()])

df = DataFrame(combined_data, columns=['posture', 'age', 'year', 'acc_x', 'acc_y', 'acc_z', 'azim', 'pitch', 'roll', 'gyro_x', 'gyro_y', 'gyro_z', 'macc_x', 'macc_y', 'macc_z', 'mazim', 'mpitch', 'mroll', 'mgyro_x', 'mgyro_y', 'mgyro_z'])
df.to_csv('data/train.csv')

train_data = read_csv('data/train.csv')

trainY = train_data['posture'].values
trainX = train_data.drop('posture', 1)

test_data = []

data_raw = raw_loader.load_eval('test1')

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
                
    age = data[' Age']

    test_data.append([posture,
                      age,
                      year,
                    data_sensor.acc_x.var(),
                    data_sensor.acc_y.var(),
                    data_sensor.acc_z.var(),
                    data_sensor.azim.var(),
                    data_sensor.pitch.var(),
                    data_sensor.roll.var(),
                    data_sensor.gyro_x.var(),
                    data_sensor.gyro_y.var(),
                    data_sensor.gyro_z.var(),
                    data_sensor.acc_x.mean(),
                    data_sensor.acc_y.mean(),
                    data_sensor.acc_z.mean(),
                    data_sensor.azim.mean(),
                    data_sensor.pitch.mean(),
                    data_sensor.roll.mean(),
                    data_sensor.gyro_x.mean(),
                    data_sensor.gyro_y.mean(),
                    data_sensor.gyro_z.mean()])

df = DataFrame(test_data, columns=['posture', 'age', 'year', 'acc_x', 'acc_y', 'acc_z', 'azim', 'pitch', 'roll', 'gyro_x', 'gyro_y', 'gyro_z', 'macc_x', 'macc_y', 'macc_z', 'mazim', 'mpitch', 'mroll', 'mgyro_x', 'mgyro_y', 'mgyro_z'])
df.to_csv('data/test.csv')

test_data = read_csv('data/test.csv')

testY = test_data['posture'].values
testX = test_data.drop('posture', 1)

rf = RandomForestClassifier(n_estimators=100, max_depth=30, warm_start=True)
rf.fit(trainX, trainY)

predTestY = rf.predict(testX)
accuracyTest = float(np.sum(predTestY == testY)) / predTestY.shape[0]
# postureTest = float(np.sum(predTestY%10 == testY%10)) / predTestY.shape[0]
# situationTest = float(np.sum(predTestY/10 == testY/10)) / predTestY.shape[0]

print "testSet accuracy: ", accuracyTest*100, "%"
# print "testSet accuracy: ", postureTest
# print "testSet accuracy: ", situationTest
