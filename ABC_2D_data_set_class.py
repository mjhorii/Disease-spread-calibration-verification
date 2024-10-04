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
    def __init__(self,data_file,params_file, mob_range,jp_range, T=300, limit_data_range = False, data_range_start = 0, data_range_end = 0, accumulated_data = False, I_ABM_ALL = None, override = False, n= None, n_time= None, n_pop= None, mobilities = None, jps = None):
        """
        data_file: file name of pickled data (I_ABM_ALL_no_acc -- new infections per time step, no accumulation)
        params_file: file name of csv file containing parameter information
        mob_range: [minimum mobility value in dataset, max mobility value in dataset]
        jp_range: [minimum jumping probability value in dataset, max jumping probability value in dataset]
        T: maximum time step/simulation length
        limit_data_range: if True, only loads data samples between [data_range_start:data_range_end], if False, loads all data samples
        data_range_start: see above
        data_range_end: see above
        accumulated_data: if True, assumes you will pass in previously calculated I_ABM_ALL (*already accumulated* data), if False, will do the accumulation calculation here
        I_ABM_ALL: to pass in previously accumulated data
        override: default False. if True, uses the inputted values for n, n_time, n_pop, mobilities, jps instead of getting them from the data
        """

        # Copy mob range
        self.min_mob=mob_range[0]
        self.max_mob=mob_range[1]
        self.min_jp=jp_range[0]
        self.max_jp=jp_range[1]

        if override == False:
            if limit_data_range == False:
                # Training data import
                with open(data_file, 'rb') as handle:
                    I_ABM_ALL_no_acc = pickle.load(handle)

                # Training mobs import
                data_params = pd.read_csv(params_file, index_col=0).to_numpy()
            else:
                # Training data import
                with open(data_file, 'rb') as handle:
                    I_ABM_ALL_no_acc = pickle.load(handle)[data_range_start:data_range_end, :]

                # Training mobs import
                data_params = pd.read_csv(params_file, index_col=0).to_numpy()[data_range_start:data_range_end, :]

            print('Data loaded')

            [self.n,self.n_time,self.n_pop]=np.shape(I_ABM_ALL_no_acc)
            self.mobilities=data_params[:,0]
            self.jps=data_params[:,1]
        else:
            self.mobilities = mobilities
            self.jps = jps
            self.n = n
            self.n_time = n_time
            self.n_pop = n_pop


        self.T=T
        self.time_vec = np.linspace(0,T,self.n_time)
        self.del_t=self.time_vec[1]-self.time_vec[0]

        if accumulated_data == False:
            # Add patient zero to first pop only
            p=0
            if I_ABM_ALL_no_acc[0,0,p]==0: # Check if the first string starts at zero, this should be the case
                I_ABM_ALL_no_acc[:,0,p]=1
            else: 
                print('Already have patient 0 in training')

            self.I_ABM_ALL=np.zeros_like(I_ABM_ALL_no_acc)

            # Accumulate over rolling window, NOW REPLACED BY CUMSUM
            print("Starting accumulation:")
            # self.I_ABM_ALL=np.cumsum(I_ABM_ALL_no_acc,axis=1)
            print(I_ABM_ALL_no_acc.shape)
            steps_to_acc = 3001
            self.steps_to_acc=steps_to_acc
            # for i in range(self.steps_to_acc):
            for i in tqdm(range(self.steps_to_acc)):
                if i==0:
                    self.I_ABM_ALL+=I_ABM_ALL_no_acc*1 # Hard copy
                else:
                    self.I_ABM_ALL[:,i:,:]+=I_ABM_ALL_no_acc[:,:-i,:]
                    #    self.I_ABM_ALL[:,i:,:] = acc(self.I_ABM_ALL, I_ABM_ALL_no_acc, i)

                # self.I_ABM_ALL = acc(self.I_ABM_ALL, I_ABM_ALL_no_acc, self.steps_to_acc)

                #self.I_ABM_ALL_no_acc = I_ABM_ALL_no_acc # I think this is not needed
        else:
            self.I_ABM_ALL = I_ABM_ALL #set to precalculated I_ABM_ALL