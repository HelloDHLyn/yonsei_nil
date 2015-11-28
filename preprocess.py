import numpy
import pandas
from pandas import *

data_raw = read_csv('data/save.csv')

# Remove useless whitespace
data_raw.rename(columns=lambda x: x.strip(), inplace=True)

# Remove unsufficient data
threshold = 100
age_useless = []

data_count = data_raw.groupby('Age').count()['WPM']
age_useless.append(data_count[data_count <= threshold].index)

for age in age_useless[0]:
    data_raw = data_raw[data_raw['Age'] != age]
