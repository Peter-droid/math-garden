# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 13:53:29 2020

@author: lx
"""
#数据来自 https://www.lixinger.com/ 数据版权属于此网站

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import theano.tensor as tt
import warnings
import csv

temp = []

filey = 'yunnanbaiyao.csv'
value_of_y = []
daily_exp_y = []


with open(filey) as y:
    reader = csv.reader(y)
    temp = list(temp)
    for i in range(101):
        value_of_y.append(float(temp[i+1][4]))
        

    
for i in range(200):
    daily_exp_y.append((value_of_y[i+1]-value_of_y[i]/value_of_y[i]))


with pm.Model() as model_y:
    mu = pm.Normal("mu", mu=0, sigma=1)
    obs = pm.Normal("obs", mu=mu, sigma=1, observed=daily_exp_y(100))
    
    sample_y = pm.sample(10000, tune=2500)
    

    
    
