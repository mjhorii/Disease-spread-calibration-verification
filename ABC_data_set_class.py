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


class data_set:
    def __init__(self,data_file,params_file, mob_range, T=300, steps_to_acc=140, limit_data_range = False, data_range_start = 0, data_range_end = 0):
        # Copy mob range
        self.min_mob=mob_range[0]
        self.max_mob=mob_range[1]

        # Training data import
        with open(data_file, 'rb') as handle:
            I_ABM_ALL_no_acc = pickle.load(handle)

        # Training mobs import
        data_params = pd.read_csv(params_file, index_col=0).to_numpy()

        # Data pre processing
        if limit_data_range == False:
            [self.n,self.n_time]=np.shape(I_ABM_ALL_no_acc)
            self.mobilities=data_params[:,0]
        else:
            [self.n,self.n_time]=np.shape(I_ABM_ALL_no_acc[data_range_start:data_range_end, :])
            self.mobilities=data_params[data_range_start:data_range_end,0]
            I_ABM_ALL_no_acc = I_ABM_ALL_no_acc[data_range_start:data_range_end, :]

        self.T=T
        self.time_vec = np.linspace(0,T,self.n_time)
        self.del_t=self.time_vec[1]-self.time_vec[0]

        # Add patient zero
        if I_ABM_ALL_no_acc[0,0]==0: # Check if the first string starts at zero, this should be the case
            I_ABM_ALL_no_acc[:,0]=1
        else: 
            print('Already have patient 0 in training')

        self.I_ABM_ALL=np.zeros_like(I_ABM_ALL_no_acc)

        # Accumulate over rolling window
        self.steps_to_acc=steps_to_acc
        for i in range(self.steps_to_acc):
            if i==0:
                self.I_ABM_ALL+=I_ABM_ALL_no_acc*1 # Hard copy
            else:
                self.I_ABM_ALL[:,i:]+=I_ABM_ALL_no_acc[:,:-i]

        self.I_ABM_ALL_no_acc = I_ABM_ALL_no_acc

class data_set_test:
    def __init__(self,num_pts,stdev=0.2,log_mode=False):
        # Copy mob range
        self.min_mob=0
        self.max_mob=1


        self.n=num_pts
        self.T=300
        self.n_time=2
        self.time_vec = np.linspace(0,self.T,self.n_time)

        self.mobilities=np.random.rand(self.n)+self.min_mob # Random from 10 to 11
        self.I_ABM_ALL=np.zeros((self.n,2))
        if log_mode:
            self.I_ABM_ALL[:,0]=self.mobilities-np.abs(np.random.lognormal(0,stdev,self.n))
        else:
            self.I_ABM_ALL[:,0]=self.mobilities+np.random.normal(0,stdev,self.n)
        self.I_ABM_ALL[:,1]=self.I_ABM_ALL[:,0]