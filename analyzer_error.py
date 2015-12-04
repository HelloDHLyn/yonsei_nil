import numpy
import pandas
from pandas import *

import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')

from dataloader import DataLoader

loader = DataLoader()

from __future__ import division

data_key = loader.get_key_data()

ascii_num = 99

raw_of_all = data_key[data_key['intent_code_point']==ascii_num]
raw_of_alphabet = data_key[data_key['intent_code_point']==ascii_num][data_key['code_point']!=ascii_num]

hands = ['Both', 'Right', 'Left']

result = []

for hand in hands:
    data_all = raw_of_all[raw_of_all['input_posture']==hand]
    data_all = data_all.groupby('keyboard_condition').count()['time']

    data_of_alphabet = raw_of_alphabet[raw_of_alphabet['input_posture']==hand]
    data_by_condition = data_of_alphabet.groupby('keyboard_condition').count()['time']

    result_both = data_by_condition[data_by_condition.index%4==0].sum() / data_all[data_all.index%4==0].sum()
    result_left = data_by_condition[data_by_condition.index%4==1].sum() / data_all[data_all.index%4==1].sum()
    result_right = data_by_condition[data_by_condition.index%4==2].sum() / data_all[data_all.index%4==2].sum()
    result_split = data_by_condition[data_by_condition.index%4==3].sum() / data_all[data_all.index%4==3].sum()
    
    result.append([result_both, result_left, result_right, result_split])
    
df = DataFrame(result, columns=['Both', 'Left', 'Right', 'Split'])
df.plot(kind='bar')
plt.show()