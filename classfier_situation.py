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
            
        situation = data[' Input situation']
        if situation == 'Sit':
            situation = 1
        elif situation == 'Stand':
            situation = 2
        elif situation == 'Walk':
            situation = 3
        else:
            situation = 0
        
        
        combined_data.append([situation,
                    data_sensor.acc_z.var(),
                    data_sensor.pitch.var(),
                    data_sensor.azim.var(),
                    data_sensor.roll.var(),
                    data_sensor.gyro_z.var(),
                    data_sensor.acc_z.mean(),
                    data_sensor.pitch.var(),
                    data_sensor.azim.mean(),
                    data_sensor.roll.mean(),
                    data_sensor.gyro_z.mean()])
        
df = DataFrame(combined_data, columns=['situation', 'acc_z', 'pitch', 'azim', 'roll', 'gyro_z', 'macc_z', 'mpitch', 'mazim', 'mroll', 'mgyro_z'])
df.to_csv('data/train1.csv')

train_data = read_csv('data/train1.csv')

trainY = train_data['situation'].values
trainX = train_data.drop('situation', 1)

test_data = []

for group in range(1, 3):
    data_raw = raw_loader.load_eval('test' + str(group))

    for idx in data_raw.index:
        data = data_raw.ix[idx]
        data_sensor = raw_loader.load_sensor_data(data['File name'])

        situation = data[' Input situation']
        if situation == 'Sit':
            situation = 1
        elif situation == 'Stand':
            situation = 2
        elif situation == 'Walk':
            situation = 3
        else:
            situation = 0

        age = data[' Age']

        test_data.append([situation,
                        data_sensor.acc_z.var(),
                        data_sensor.pitch.var(),
                        data_sensor.azim.var(),
                        data_sensor.roll.var(),
                        data_sensor.gyro_z.var(),
                        data_sensor.acc_z.mean(),
                        data_sensor.pitch.var(),
                        data_sensor.azim.mean(),
                        data_sensor.roll.mean(),
                        data_sensor.gyro_z.mean()])

df = DataFrame(test_data, columns=['situation', 'acc_z', 'pitch', 'azim', 'roll', 'gyro_z', 'macc_z', 'mpitch', 'mazim', 'mroll', 'mgyro_z'])
df.to_csv('data/test1.csv')

test_data = read_csv('data/test1.csv')

testY = test_data['situation'].values
testX = test_data.drop('situation', 1)

rf = RandomForestClassifier(n_estimators=100, max_depth=30, max_leaf_nodes=100)
rf.fit(trainX, trainY)

predTestY = rf.predict(testX)
accuracyTest = float(np.sum(predTestY == testY)) / predTestY.shape[0]

print "testSet accuracy: ", accuracyTest*100, "%"