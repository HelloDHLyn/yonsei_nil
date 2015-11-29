import numpy
import pandas
from pandas import *

class RawDataLoader:
    def __init__(self):
        self.data_path = ''

    def load_eval(self, name):
        self.data_path = 'data/raw/' + name + '/'
        eval_path = self.data_path + 'save_evaluation.csv'
        return read_csv(eval_path)
        
    def load_sensor_data(self, filename):
        file_path = self.data_path + 'Sensor/Sensor_' + filename + '.csv'
        return read_csv(file_path)
        

class DataLoader:
    data_save = None
    data_key = None
    data_sensor = None
    
    threshold = 100
    age_useless = []
    
    def __init__(self):
        data_save_raw = read_csv('data/save.csv')
        data_key_raw = read_csv('data/key.csv')
        data_sensor_raw = read_csv('data/sensor.csv')
        
        # Remove useless whitespace
        self.data_save = data_save_raw.rename(columns=lambda x: x.strip().lower())
        self.data_key = data_key_raw.rename(columns=lambda x: x.strip().lower())
        self.data_sensor = data_sensor_raw.rename(columns=lambda x: x.strip().lower())

        # Remove unsufficient data
        data_count = self.data_save.groupby('age').count()['wpm']
        self.age_useless.append(data_count[data_count <= self.threshold].index)

    def get_save_data(self):
        return self.remove_useless(self.data_save)
    
    def get_key_data(self):
        return self.remove_useless(self.data_key)
    
    def get_sensor_data(self):
        return self.remove_useless(self.data_sensor)
    
    def remove_useless(self, data):
        for age in self.age_useless[0]:
            data = data[data['age'] != age]
            
        return data