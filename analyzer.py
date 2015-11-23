import numpy
import pandas
from pandas import *

import matplotlib
import matplotlib.pyplot as plt

data = read_csv('data/save.csv')

# Set columns
wpm_per_age = data.groupby(' Age').mean()[' WPM']
print wpm_per_age

wpm_per_posture = data.groupby(' Input posture').mean()[' WPM']
print wpm_per_posture

matplotlib.style.use('ggplot')

# Line Graph
plt.plot(wpm_per_age.index, wpm_per_age)

# Bar Graph
# wpm_per_posture.plot(kind='bar')

plt.show()