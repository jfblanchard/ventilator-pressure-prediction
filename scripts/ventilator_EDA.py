# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 10:29:35 2021

Ventilator prediction working

@author: Jon
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# load files
train = pd.read_csv(r'F:\Sync\Work\Kaggle Competitions\Ventilator Pressure Prediction\Data\train.csv')
test = pd.read_csv(r'F:\Sync\Work\Kaggle Competitions\Ventilator Pressure Prediction\Data\test.csv')

# function to get single breath (id must exist...there are some that are missing)
def get_breath(df,id):
    # generate start stop points
    start = 80 * id
    stop = start + 80
    return train.iloc[start:stop,:]

# Get a single breath
myid = 0
id1 = get_breath(train,myid)
lag = id1.u_in.shift(1,fill_value = 0)
id1['lag'] = lag

# make a dt column
dt = np.diff(id1.time_step)
dt_list = list(dt)
dt_list.append(dt.mean())
d_uin = np.diff(id1.lag)
d_pressure = np.diff(id1.pressure)
id1['dt'] = dt_list

# make a volume column from u_in 
vol = np.zeros(len(id1))
vol[0] = 6  # start with avg starting value of pressure
r = id1.R.iloc[0]
c = id1.C.iloc[0]
for i in range(len(id1)-1):
    if id1.u_out.iloc[i] == 0:
        vol[i+1] = vol[i] + (id1.dt.iloc[0]*id1.lag.iloc[i+1] * (c/r)) *-.3*np.log(id1.time_step.iloc[i+1]*1)
    else:
        vol[i+1] == 6


# plot a single breath
pal1 = sns.color_palette("viridis",3)
sns.set_palette(pal1)


r = id1.R.iloc[0]
c = id1.C.iloc[0]
plt.figure(figsize=(8,5))
plt.plot(id1.pressure,label='pressure')
plt.plot(id1.lag,label='u_in')
plt.plot(id1.u_out,label='u_out')
plt.plot(vol,label='vol',color='red')
plt.title(f'Pressure and u_in for Breath id={id}, R={r}, C={c}')
plt.legend();