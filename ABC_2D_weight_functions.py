
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math
import time
from tqdm import tqdm
import cProfile
import pickle
import scipy as sci
import pandas as pd

# Weight functions
def norm_weights(func,param,scores,standardized=1): # Standardized should be 1 or zero
    #args_arr=np.array(transform(list(scores.flatten()))).reshape(np.shape(scores))
    args_arr=scores.argsort(axis=None).argsort(axis=None).reshape(np.shape(scores)) # This is the ranks
    weights=func(param,scores,args_arr,standardized)
    return weights/np.sum(weights)

def step(paramS,scores,args_arr,standardized=0): #c is cut off percentage
    param=paramS*1
    if standardized: param*=1
    return (args_arr<=np.size(args_arr)*(param))*1.0

def neg_exp(paramS,scores,args_arr,standardized=0): #c is 1/ multiplier
    param=paramS*1
    if standardized: param*=1/2
    return np.e**(-1/param*args_arr/np.size(args_arr))

def linear(paramS,scores,args_arr,standardized=0): #c is x intercept
    param=paramS*1
    if standardized: param*=1/2*3
    w=-1/param*(args_arr/np.size(args_arr))+1 #Normalize and apply linear
    return np.abs(w*(w>0)) # Return only possives, abs for no weird negative zeros

def epanechnikov(paramS,scores,args_arr,standardized=0): #c is delta, set the kernel's other c to 1 because we normalize anyway
    param=paramS*1
    if standardized: param*=1/2/3*8
    nor_args_arr=args_arr/np.size(args_arr) # Normalize rank
    w=1/param*(1-(nor_args_arr/param)**2) #Normalize and apply kernel
    return np.abs(w*(w>0)) # Return only possives, abs for no weird negative zeros