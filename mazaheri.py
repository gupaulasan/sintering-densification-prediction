# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 14:30:04 2023

@author: gupaulasan
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import time
import math

pd.set_option("display.precision", 20)

def read_clean_csv(file, double_headed = True, x='tempC', y='fracDensity'):
    '''
    Reads and cleans a csv imported from WebPlotDigitilizer

    file: csv file imported from WebPlotDigitilizer.
    double_headed: weather the csv has more two headers (e.g. 2Kmin and [x,y])
    x: name of the columns that represents the x axis
    y: name of the columns that represents the y axis
    '''
    if double_headed:
        df = pd.read_csv(file, header=[0,1])
        new_cols = [list(tup) for tup in df.columns.values]
        for i in range(len(df.columns.values)):
            if i % 2 != 0:
                new_cols[i][0] = new_cols[i-1][0]
                new_cols[i][1] = y
            else:
                new_cols[i][1] = x
        columns = [tuple(lis) for lis in new_cols]
        df.columns = columns
        df.columns = ['_'.join(col) for col in df.columns.values]
    else:
        df = pd.read_csv(file,header=None)
        df.columns = [x,y]
    return df

def model_gomez(df, beta, Ea, A, n, G, R = 8.31446262, beta_SI = False):
    '''
    Uses the data to predict the densification in acordance to the Gomez-Hotza model
    
    df: clean data from WebPlotDigitizer
    beta: heating rate of the process. Either in °C(K)/min or °C(K)/s
    Ea: material's activation energy
    A: Gomez-Hotza coefficient
    n: mass transport coefficiente, varies from 0.0 to 1.0
    G: mean particle size
    R: molar gas constant
    beta_SI: weather the beta value is expressed in °C(K)/s or not
    '''
    #df = df1.copy
    if not beta_SI:
        beta_sec = beta/60
    else:
        beta_sec = beta
    df['tempK'] = df['tempC'] + 273.15
    df['theta'] = 1 - df['fracDensity']
    df['ln_theta_theta0'] = np.log(df['theta']/df['theta'][0])
    df['t'] = (df['tempK'] - df['tempK'][0])/beta_sec
    df['tn'] = df['t'] ** n
    df['Tminus1'] = 1/df['tempK']
    df['exp_ERT'] = np.exp(-((Ea)/(R *df['tempK'])))
    df['tnj_tnj1'] = np.zeros(len(df['tn']))
    for i in range(len(df['tn'])-1):
        df['tnj_tnj1'][i] =  (df['tn'][i+1] - df['tn'][i])
    df['pred'] = (A/G) * (df['Tminus1'] * df['tnj_tnj1'] * df['exp_ERT'])
    df['abs_error'] = np.abs(df['pred'] - df['ln_theta_theta0'])
    df['squared_error'] = np.square(df['pred'] - df['ln_theta_theta0'])
    return df

def mae(df, column):
    '''
    Computes the mean absolute error of a column in a DataFrame
    
    df: DataFrame
    column: string representig the name of the column in whiche the absolute error is expressed
    '''
    mean_abs_error = np.mean((df[column]))
    return mean_abs_error

def mse(df, column):
    '''
    Computes the mean squared error of a column in a DataFrame
    
    df: DataFrame
    column: string representig the name of the column in whiche the squared error is expressed
    '''
    mean_squared_error = np.mean((df[column]))
    return mean_squared_error

def model_validation(df, A_values, Ea_values, n_values, beta, G, metric = 'mse',R = 8.31446262, beta_SI = False, print_iteration = False ):
    '''
    Evaluates the parameters A, Ea and n of the Gomez-Hotza model based-off a Grid Search of parameters
    
    df: DataFrame
    A_values: list of values of A to be tested
    Ea_values: list of values of Ea to be tested
    n_values: list of values of n to be tested
    beta: heating rate of the process. Either in °C(K)/min or °C(K)/s
    G: mean particle size
    metric: either 'mse' (mean squared error) or 'mae'(mean absolute error), it refers to the error parameter used
    R: molar gas constant
    beta_SI: weather the beta value is expressed in °C(K)/s or not
    print_iteration: when True the program prints the chosen error for each iteration
    '''
    st = time.time()
    best_error = float('inf')
    list_of_dict = []
    for A in A_values:
        for Ea in Ea_values:
            for n in n_values:
                data = model_gomez(df, beta, Ea, A, n, G)
                if metric == 'mse':
                    error = mse(data, 'squared_error')
                    list_of_dict.append({'A':A, 'Ea': Ea, 'n':n,'MSE':error})
                    if print_iteration:
                        print(f'MSE para [{A},{Ea}, {n}] = {error}')
                    if error < best_error:
                        best_error = error
                        best_params = {'A':A, 'Ea': Ea, 'n':n}
                else:
                    error = mae(data, 'abs_error')
                    list_of_dict.append({'A':A, 'Ea': Ea, 'n':n,'MAE':error})
                    if print_iteration:
                        print(f'MSE para [{A},{Ea}, {n}] = {error}')
                    if error < best_error:
                        best_error = error
                        best_params = {'A':A, 'Ea': Ea, 'n':n}
    print(f'Best {metric}     :{best_error}')
    print(f'Best params  :{best_params}')
    end = time.time() - st
    print(f'Run time: {end}s')
    return list_of_dict

mazaheri = read_clean_csv("C:/Users/gupau/Documents/UFSC - PC/Semestre 23.2/TCC/Mazaheri/mazaheri.csv")
print(mazaheri.head())

mazaheri_2K = pd.DataFrame(mazaheri[mazaheri.columns.values[0:2]])
mazaheri_5K = pd.DataFrame(mazaheri[mazaheri.columns.values[2:4]])
mazaheri_20K = pd.DataFrame(mazaheri[mazaheri.columns.values[4:]])

mazaheri_2K.columns = [col[6:] for col in mazaheri_2K.columns.values]
mazaheri_5K.columns = [col[6:] for col in mazaheri_5K.columns.values]
mazaheri_20K.columns = [col[7:] for col in mazaheri_20K.columns.values]

validation2K = pd.DataFrame(model_validation(mazaheri_2K, np.linspace(100, 50, 10), np.linspace(500000,700000,10), [0.2, 0.3, 0.4, 0.5], beta=2, G=7.5e-8))
validation5K = pd.DataFrame(model_validation(mazaheri_5K, np.linspace(100, 50, 10), np.linspace(500000,700000,10), [0.2, 0.3, 0.4, 0.5], beta=5, G=7.5e-8))
validation20K = pd.DataFrame(model_validation(mazaheri_20K, np.linspace(100, 50, 10), np.linspace(500000,700000,10), [0.2, 0.3, 0.4, 0.5], beta=20, G=7.5e-8))

g = sns.FacetGrid(validation2K, col='Ea', hue='n', col_wrap=5)
g.map(sns.scatterplot, 'A', 'MSE', alpha=0.5)
g.add_legend()
plt.show()
plt.clf()

g = sns.FacetGrid(validation5K, col='Ea', hue='n', col_wrap=5)
g.map(sns.scatterplot, 'A', 'MSE', alpha=0.5)
g.add_legend()
plt.show()
plt.clf()

g = sns.FacetGrid(validation20K, col='Ea', hue='n', col_wrap=5)
g.map(sns.scatterplot, 'A', 'MSE',alpha=0.5)
g.add_legend()
plt.show()
plt.clf()