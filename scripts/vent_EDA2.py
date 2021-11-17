# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 10:12:01 2021

@author: Jon
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
pal1 = sns.color_palette("RdBu",7)
sns.set_palette(pal1)

# load files -----------------------------------------------------------------
vpath = r"F:\Sync\Work\Kaggle Competitions\Ventilator Pressure Prediction\Data\\"
train = pd.read_csv(vpath + 'train.csv')
test = pd.read_csv(vpath + 'test.csv')

# determine valid breath ids --------------------------------------------------
train_bids = train.breath_id.unique()
test_bids = test.breath_id.unique()

# add any features
train['max_times'] = train.groupby('breath_id').time_step.transform('max')
train['dt'] = train.groupby('breath_id').time_step.transform('max')/80


train['R*C'] = train['R'] * train['C']

#train['dt'] = np.diff(train.time_step)

# separate into inhale and exhale sections
train_in = train[train.u_out == 0]
train_ex = train[train.u_out == 1]
test_in = test[test.u_out == 0]



#b_in['dt'] = np.diff(b_in.time_step)


for shift in range(4):
    
    # look at individual breaths
    bid = 5
    b_in = train_in[train_in.breath_id == train_bids[bid]]
    b_ex = train_ex[train_ex.breath_id == train_bids[bid]]

    b_in.u_in = b_in.u_in.shift(shift,fill_value=0)
    
    # plot u_in vs pressure
    plt.figure(figsize=(9,6))
    plt.plot(b_in['u_in'], label='u_in')
    plt.plot(b_in['pressure'],label=['pressure'])
    plt.legend()
    plt.title(f'U_in and Pressure for breath_id={train_bids[bid]}, shift = {shift}')
    plt.savefig(f'train_bid_{train_bids[bid]}',dpi=300)
    
    
    # run a linear regression model 
    from sklearn.linear_model import LinearRegression
    
    y = b_in.pressure
    b1_prep = b_in.drop('pressure',axis=1)
    b1_prep = b1_prep.drop('id',axis=1)
    
    model_lr = LinearRegression()
    model_lr.fit(b1_prep,y)
    preds_lr = model_lr.predict(b1_prep)
    residuals = np.abs(y -preds_lr)
    
    plt.figure(figsize=(10,6))
    plt.plot(y.values, label='y')
    plt.plot(preds_lr, label='preds')
    plt.plot(residuals.values, label='residuals')
    plt.title(f'Linear Regression on breath_id={train_bids[bid]}, shift = {shift}, MAE = {np.mean(residuals)}')
    plt.legend()
    plt.savefig(f'Linear regression bid_{train_bids[bid]}-shift={shift}',dpi=300)
    
    print(f'Mean absolute error: {np.mean(residuals)}')  #identical
    print(f'mae2 = {np.sum(residuals)/len(residuals)}')

# shift 0 MAE = .894
# shift 1 MAE = .545
# shift 2 MAE = .379
# shift 3 MAE = .628

# do it again with breath 2

# shift 0 MAE = .324
# shift 1 MAE = .531
# shift 2 MAE = .452
# shift 3 MAE = .463

# breath 3, best is shift 2
# breath 4, best is shift 0
# breath 6, best is shift 3

# other scraps _________________________________________________________________

# what is varation in time?

# max_times = train.groupby('breath_id').time_step.max()
# plt.figure()
# plt.plot(max_times)
# plt.title("Maximum Times in Seconds by breath_id")
# plt.figure()
# intervals = max_times/80
# plt.plot(intervals)

# max_times[max_times > 2.8], consider removing these
# breath_id
# 36175     2.934589
# 38415     2.905639
# 44245     2.937238
# 55851     2.936345
# 74766     2.899221
# 109693    2.905298
# 111439    2.928005

