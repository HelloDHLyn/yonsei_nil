import numpy
import pandas
from pandas import *

import matplotlib
import matplotlib.pyplot as plt

from preprocess import DataLoader

loader = DataLoader()
data = loader.get_save_data()

# Set columns
wpm_per_age = data.groupby('age').mean()['wpm']
print wpm_per_age

matplotlib.style.use('ggplot')

# Line Graph
plt.plot(wpm_per_age.index, wpm_per_age)

# Bar Graph
# wpm_per_posture.plot(kind='bar')

plt.show()
