from __future__ import print_function, division
from math import sqrt as ssqrt
from math import cos as scos
from math import fabs as fabs     # more restrictive than abs() (in a good way)
from numpy import *
from NuCraft import *
from scipy import interpolate, integrate
from scipy.stats import lognorm
from ast import literal_eval
import matplotlib.pyplot as plt
import numpy as np
from os import path

import os, sys
from time import time
from sys import stdout
set_printoptions(precision=5, linewidth=150)


def bin_matrix(matrix, eList):
    
    
    # Binning
    cos_theta_binning = np.linspace(-1, 1, 80)

    # cos intervals for binning
    cos_intervals = np.round(np.arange(1, -1, -0.1), 2)
    cos_bins = np.digitize(cos_theta_binning, cos_intervals, right=True)                              

    # energy intervals
    energy_min = np.genfromtxt('numu_cc.csv', skip_header = 1, delimiter =',')[:,0]
    energy_max = np.genfromtxt('numu_cc.csv', skip_header = 1, delimiter =',')[:,1]
    energy_bins = np.digitize(eList, energy_min)
    energy_binwidths = energy_max - energy_min
    
    #Binning
    matrix_ybinned = np.zeros((len(cos_intervals) , len(matrix[0])))

    # averaging rows in same interval
    for i in range(1, len(cos_intervals)+1):
        rows = np.where(cos_bins == i)[0]
        averaged_row = np.mean(matrix[rows[0] : rows[-1]+1, : ], axis = 0)
        matrix_ybinned[ i-1 , : ] = averaged_row
    
    binned_matrix = np.zeros((len(cos_intervals), len(energy_min)))  

    # averaging columns in same interval
    for i in range(1, len(energy_min)+1):
        columns = np.where(energy_bins == i)[0]
        averaged_column = np.mean(matrix_ybinned[ : , columns[0] : columns[-1]+1], axis = 1)
        binned_matrix[ : , i-1 ] = averaged_column
        
    return binned_matrix


def oscillation_prob(theta23, DM31):

    pType = 14   # numu
    
    max = amax
    min = amin

    eBins = 50
    # energy range in GeV
    eList = np.logspace(0, 2, eBins)
    
    n_theta = 56   # 80 in the total cos-range
    cos_theta = np.linspace(-1, 0.4, n_theta)
    zList = np.arccos(cos_theta)
    
    # parameters from http://www.nu-fit.org/?q=node/238
    # he is also using the approximation DM31 = DM32
    DM21   = 7.42e-5
    sin_squared_13 = 0.02246
    sin_squared_12 = 0.304
    theta13 = arcsin(sqrt(sin_squared_13))/pi*180.
    theta12 = arcsin(sqrt(sin_squared_12))/pi*180.

    AkhmedovOsci = NuCraft((1., DM21, DM31), [(1,2,theta12),(1,3,theta13,0),(2,3,theta23)])
    
    atmosphereMode = 3 
    numPrec = 5e-4

    zListLong, eListLong = meshgrid(zList, eList)
    zListLong = zListLong.flatten()
    eListLong = eListLong.flatten()
    tListLong = ones_like(eListLong)*pType

    # actual call to nuCraft for weight calculations:
    prob = AkhmedovOsci.CalcWeights((tListLong, eListLong, zListLong), numPrec=numPrec, atmMode=atmosphereMode)

    prob = rollaxis(array(prob).reshape(len(eList), len(zList),-1), 0,3)
    
    
    prob_numu_to_nutau = np.zeros((80, eBins))
    prob_numu_to_numu = np.zeros((80, eBins))
    prob_numu_to_nue = np.zeros((80, eBins))

    for i in range(80):
        if i>=56:
            prob_numu_to_nue[i] = 1
            prob_numu_to_numu[i] = 1
            prob_numu_to_nutau[i] = 1
        else:
            prob_numu_to_nue[i] = prob[i][0]
            prob_numu_to_numu[i] = prob[i][1]
            prob_numu_to_nutau[i] = prob[i][2]
        
    binned_prob = bin_matrix(prob_numu_to_numu, eList)
    
    return prob_numu_to_numu, binned_prob