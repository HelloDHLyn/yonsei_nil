import numpy
import pandas
from pandas import *

data_save_raw = read_csv('data/save.csv')
data_key_raw = read_csv('data/key.csv')
data_sensor_raw = read_csv('data/sensor.csv')

# Remove useless whitespace
data_save = data_save_raw.rename(columns=lambda x: x.strip().lower())
data_key = data_key_raw.rename(columns=lambda x: x.strip().lower())
data_sensor = data_sensor_raw.rename(columns=lambda x: x.strip().lower())

# Remove unsufficient data
threshold = 100
age_useless = []

data_count = data_save.groupby('age').count()['wpm']
age_useless.append(data_count[data_count <= threshold].index)

for age in age_useless[0]:
    data_save = data_save[data_save['age'] != age]
    data_key = data_key[data_key['age'] != age]
    data_sensor = data_sensor[data_sensor['age'] != age]
