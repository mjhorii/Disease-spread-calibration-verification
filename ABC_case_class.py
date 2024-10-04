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
import matplotlib.colors as mcolors

from ABC_weight_functions import * # Just for pyrolance, does not actually do the import

class case_set:
    def __init__(self,sample_data,training_data):
        self.sample_data=sample_data
        self.training_data=training_data
        self.case_list=[]

    def initialize_all_cases(self,mod=1): # Use the mod if you want to test with only every Xth case
        self.case_list=[]
        print('Making case list out of 1 in '+str(mod)+' curves in the sample data')
        for s_idx in range(self.sample_data.n):
            if s_idx%mod==0:
                self.case_list.append(case_obj(sample_curve=self.sample_data.I_ABM_ALL[s_idx,:],true_mob=self.sample_data.mobilities[s_idx],training_data=self.training_data))

    def run_scoring(self):
        print('Scoring all cases:')
        for case in tqdm(self.case_list):
            case.update_scores()

    def run_single_analysis(self,weight_function,centriod,estimator_bw,res=1000):
        print('Running single analysis for all case in list')
        for i,case in enumerate(self.case_list):
            print('Running single analysis for case: '+str(i))
            case.make_single_analysis(weight_function,centriod,estimator_bw,res)
    
    def run_analysis_array(self,weight_function_vect,centriod_vect,estimator_bw_vect,res=1000):
        print('Running analysis grid for all case in list')
        for i,case in enumerate(self.case_list):
            print('Running array analysis for case: '+str(i))
            case.make_analysis_array(weight_function_vect,centriod_vect,estimator_bw_vect,res,verbos=False)

    def get_attribute_from_case_list(self,desired_attribute):
        temp_data_list=[]
        if hasattr(self.case_list[0],desired_attribute): # Check if the attribute name is valid
            for i,case in enumerate(self.case_list): # If so, loop through and collect the data
                temp_data_list.append(getattr(case,desired_attribute))
            
            return temp_data_list

        else:
            print('Case does not have attribute: '+desired_attribute)
            return 0
    
    def get_attribute_from_case_list_analysis_array(self,desired_attribute): # Returns a list of arrays with the requested data
        temp_data_list=[]
        for i,case in enumerate(self.case_list): # Loop through and collect the data
            temp_data_list.append(case.get_attribute_from_array(desired_attribute))
            
        return temp_data_list
    

class case_obj:
    def __init__(self,sample_curve,true_mob,training_data):
        self.sample_curve=sample_curve
        self.true_mob=true_mob
        self.training_data=training_data # Pass by reference
        self.analysis_list=[]

    def update_scores(self):
        #self.scores=np.zeros(self.training_data.n)
        self.scores=np.sum((self.training_data.I_ABM_ALL - np.expand_dims(self.sample_curve,axis=[0]))**2,axis=1)**0.5 #Add 2 extra dims to the single curve to make it project

    def make_single_analysis(self,weight_function,centriod,estimator_bw,res=1000):
        self.single_res=res
        self.analysis_single=analysis_obj(self,weight_function,centriod,estimator_bw,res)
        self.analysis_list.append(self.analysis_single)

    def make_analysis_array(self,weight_function_vect,centriod_vect,estimator_bw_vect,res=1000,verbos=True):
        self.weight_function_vect=weight_function_vect
        self.centriod_vect=centriod_vect
        self.estimator_bw_vect=estimator_bw_vect
        self.res_array=res

        # Initialize arrays
        [self.weight_function_array,self.centriod_array,self.estimator_bw_array] = np.meshgrid(self.weight_function_vect,self.centriod_vect,self.estimator_bw_vect, indexing='ij')
        self.analysis_array=np.empty(np.shape(self.weight_function_array),dtype=object) # This will store pointers to the analysis objects

        # Make each analysis
        for i,weight_function in enumerate(self.weight_function_vect):
            for j,centriod in enumerate(self.centriod_vect):
                for k,estimator_bw in enumerate(self.estimator_bw_vect):
                    if verbos:
                        print('Performing analysis w function: '+weight_function.__name__+' -- cent: '+str(centriod)+' -- bw: '+str(estimator_bw))
                    self.analysis_array[i,j,k]=analysis_obj(self,weight_function,centriod,estimator_bw, res=res)

        # Make unraveled version of the pointer array
        self.analysis_array_ravel=self.analysis_array.ravel()

    def get_attribute_from_array(self,desired_attribute):
        temp_data_list=[]
        if hasattr(self.analysis_array_ravel[0],desired_attribute): # Check if the attribute name is valid
            for i,analysis in enumerate(self.analysis_array_ravel): # If so, loop through and collect the data
                temp_data_list.append(getattr(analysis,desired_attribute))
            
            return np.reshape(np.array(temp_data_list),np.shape(self.analysis_array)+np.shape(temp_data_list[1])) # Reshape the data into a grid

        else:
            print('Analysis grid does not have attribute: '+desired_attribute)
            return 0
        
class analysis_obj():
    def __init__(self,case,weight_function,centriod,estimator_bw,res=1000):
        self.case=case
        self.weight_function=weight_function
        self.centriod=centriod
        self.estimator_bw=estimator_bw
        self.res=res

        # Do the analysis
        self.update_KDE()
        self.update_analysis()

    def update_KDE(self):
        self.grid=np.linspace(self.case.training_data.min_mob,self.case.training_data.max_mob,self.res)
        self.grid_mids=mids(self.grid) # grid is at midpoints
        self.KDE=np.zeros_like(self.grid_mids) # KDE defined at mid points
        self.weights=norm_weights(self.weight_function,self.centriod,self.case.scores)
        self.data_std = np.sqrt(np.cov(self.case.training_data.mobilities,rowvar = True, bias = False, aweights = self.weights))
        kde_std = self.data_std*self.estimator_bw #scale data std by estimator bw
        # Build KDE by summing the gaussians
        for weight,mob in zip(self.weights,self.case.training_data.mobilities):
            if weight>0:
                new_vals=get_guassian_on_array(self.grid_mids,mu=mob,sig=kde_std)
                # Normalize
                new_vals=new_vals/np.sum(new_vals)*weight
                # Add to KDE
                self.KDE+=new_vals
        # Normalize over all KDE
        self.KDE=self.KDE/np.sum(self.KDE)

    def update_analysis(self):
        # Make CDF
        self.CDF=np.zeros_like(self.grid)
        self.CDF[1:]=np.cumsum(self.KDE) # Leave the first term as zero
        self.CDF_interp=sci.interpolate.interp1d(self.grid, self.CDF, kind='linear') # Input is mob, output is probability
        self.inv_CDF_interp=sci.interpolate.interp1d(self.CDF,self.grid, kind='linear') # Input is value 0 to 1

        # Post processing
        self.range_95=np.array([self.inv_CDF_interp(0.025),self.inv_CDF_interp(1-0.025)])
        self.range_50=np.array([self.inv_CDF_interp(0.25),self.inv_CDF_interp(0.75)])
        self.mean=self.inv_CDF_interp(0.5)
        self.range_size_95=np.diff(self.range_95)
        self.range_size_50=np.diff(self.range_50)
        self.continous_rank=self.CDF_interp(self.case.true_mob)

        # Non-kde rank
        self.match_idx=np.where(self.weights>0,1,0)
        # What fraction of the time was a match less than the true mob?
        self.rank=np.sum(self.match_idx * np.where(self.case.training_data.mobilities<self.case.true_mob,1,0))/np.sum(self.match_idx)

        #Check how many matches are included (non-zero weights)
        self.non_zero_matches = np.sum(np.where(self.weights>0,1,0))

        #Calculate continuous ranked probability score
        d_mob = (self.case.training_data.max_mob-self.case.training_data.min_mob)/self.res
        self.crps = d_mob*np.sum((self.CDF-(self.grid>=self.case.true_mob))**2)

        # Pass?
        self.range_pass_95=(self.range_95[0]<=self.case.true_mob and self.range_95[1]>=self.case.true_mob)
        self.range_pass_50=(self.range_50[0]<=self.case.true_mob and self.range_50[1]>=self.case.true_mob)

    def make_kde_plot(self,title='KDE', legend = True, plot_matches=False):

        #------------------ Plot sample KDE w CIs, true mobility -----------------
        KDE = self.KDE

        mob_value = self.case.true_mob

        upper_percentile_95 = self.range_95[0]
        lower_percentile_95 = self.range_95[1]
        upper_percentile_50 = self.range_50[0]
        lower_percentile_50 = self.range_50[1]

        grid_mids=self.grid_mids # grid is at midpoints

        plt.figure(dpi = 500, figsize = (4,3))
        plt.plot(grid_mids,KDE, label = 'ABC Posterior')

        plt.axvline(upper_percentile_95, color = 'red', label = '95% CI')
        plt.axvline(lower_percentile_95, color = 'red')
        plt.axvline(upper_percentile_50, color = 'gold', label = '50% CI')
        plt.axvline(lower_percentile_50, color = 'gold')
        plt.axvline(mob_value, color = 'limegreen', label = 'True mobility', linestyle = '--')

        # Plot matches
        if plot_matches == True:
            colors = [(1,0,0,c) for c in np.linspace(0,1,100)]
            cmapblue = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=100)
            plt.scatter(self.case.training_data.mobilities,np.ones_like(self.case.training_data.mobilities)*max(KDE)*0.05,marker='x',c = self.weights, cmap = cmapblue,s=50)

            # Use empty lines to get legend
            d = []
            plt.scatter(d, d, marker='x', color=(1,0,0), label = 'Matches')

        plt.xlabel('Mobility')
        plt.ylabel('Posterior density')
        plt.legend()
        plt.ylim(0, max(KDE)*1.2)
        plt.title(title)
        # plt.savefig('quals_figs/ABC_results_1d_KDE_for_sample_'+str(sample_ind)+'_centroid_ind_'+str(centroid_ind)+'_bw_ind_'+str(bw_ind)+'.png', dpi=500, bbox_inches='tight') 
        plt.show()

####### Helper functions ############

def get_guassian_on_array(grid,mu,sig): # mu is mean sig is std
    return 1/(sig*(2*np.pi)**0.5)*np.e**(-0.5*((grid-mu)/sig)**2)

def mids(x):
    return (x[1:]+x[:-1])/2