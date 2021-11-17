# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 09:23:24 2021

@author: Jon

Writing a simplified version of the R_spring constant

"""
import numpy as np
import matplotlib.pyplot as plt


R = [5, 20, 50]
C = [10, 20, 50]
t = np.arange(0,2.8,.035)
max_t = np.max(t)
spring = (1- np.exp(-t/max_t))


# plot out functions
plt.figure(figsize=(10,6))
plt.title(f'Spring function for R = 5,20,50')
for r in R:
    plt.plot(t,r*spring, label = r)
plt.legend()