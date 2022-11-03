import matplotlib.pyplot as plt
import numpy as np
from numba import njit
import sys
from scipy import stats
from scipy.stats import binom, poisson, norm
from iminuit import Minuit
from sympy.tensor.array import derive_by_array
from numpy import identity, array, dot, matmul
from latex2sympy2 import latex2sympy
from sympy import *
from matplotlib import cm, colors
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append('C:\\Users\\Bruger\\AppStat2021\\External_Functions')
import Clotilde_external_functions as cef
import ExternalFunctions as ef
from ExternalFunctions import nice_string_output, add_text_to_ax, UnbinnedLH



def bin_matrix(matrix, eList):
    
    
    # Binning
    cos_theta_binning = np.linspace(-1, 1, 200)

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



def calcPathLength(cz,r=6371.,h=15.,d=1.) :
    '''
    cz = cos(zenith) in radians, to be converted to path length in km
    r = Radius of Earth, in km
    h = Production height in atmosphere, in km
    d = Depth of detector, in km
    '''
    return -r*cz +  np.sqrt( (r*cz)**2 - r**2 + (r+h+d)**2 )



def prob_numu_numu(theta23, DM32):
    
    
    eBins = 200
    eList = np.linspace(1, 100, eBins)  # GeV
    
    n_theta = 200
    cos_theta = np.linspace(-1, 1, n_theta)

    # parameters from //www.nu-fit.org/?q=node/238. We use the approximation DM31 = DM32
    #DM21   = 7.60e-5   #eV
    theta13 =  8.62/180 * np.pi
    theta12 = 33.45/180 * np.pi 
    
        
    L = calcPathLength(cos_theta) # km
    prob = 1 - 4*np.sin(theta23)**2 *np.cos(theta13)**2 *(1 - np.sin(theta23)**2 *np.cos(theta13)**2) \
    *np.sin(-1.27 * DM32 * L[np.newaxis,:].T /eList)**2
    
    binned_prob = bin_matrix(prob, eList)
    
    
    return prob, binned_prob
