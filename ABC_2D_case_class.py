import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math
import time
from tqdm import tqdm
#import cProfile
import pickle
import scipy as sci
import pandas as pd
import matplotlib.colors as mcolors
import numba as nb

from ABC_weight_functions import * # Just for pyrolance, does not actually do the import

class case_set:
    def __init__(self,sample_data,training_data):
        self.sample_data=sample_data
        self.training_data=training_data
        self.case_list=[]

    def initialize_all_cases(self,mod=1,no_data=False): # Use the mod if you want to test with only every Xth case, use no_data = True if you dont have any I_ABM_ALL and only want to upload parameter values
        self.case_list=[]
        print('Making case list out of 1 in '+str(mod)+' curves in the sample data')
        if no_data == False:
            for s_idx in range(self.sample_data.n):
                if s_idx%mod==0:
                    self.case_list.append(case_obj(sample_curve=self.sample_data.I_ABM_ALL[s_idx,:,:],true_mob=self.sample_data.mobilities[s_idx],true_jp=self.sample_data.jps[s_idx],training_data=self.training_data))
        else:
            for s_idx in range(self.sample_data.n):
                if s_idx%mod==0:
                    self.case_list.append(case_obj(sample_curve=None,true_mob=self.sample_data.mobilities[s_idx],true_jp=self.sample_data.jps[s_idx],training_data=self.training_data))

    def run_scoring(self, style = "concat"):
        print('Scoring all cases:')
        for case in tqdm(self.case_list):
            case.update_scores(style = style)

    def set_pre_calculated_scores(self, scores): 
        #scores should be a list in the format: [case_set.case_list[0].scores, case_set.case_list[1].scores, ...]
        for i,case in enumerate(self.case_list):
            case.scores = scores[i]

    def run_single_analysis(self,weight_function,centriod,estimator_bw,res=100):
        print('Running single analysis for all case in list')
        for i,case in enumerate(self.case_list):
            print('Running single analysis for case: '+str(i))
            case.make_single_analysis(weight_function,centriod,estimator_bw,res)
    
    def run_analysis_array(self,weight_function_vect,centriod_vect,estimator_bw_vect,res=100):
        print('Running analysis grid for all case in list')
        for i,case in enumerate(self.case_list):
            print('Running array analysis for case: '+str(i))
            case.make_analysis_array(weight_function_vect,centriod_vect,estimator_bw_vect,res,verbos=False)

    def run_partial_analysis_array(self,weight_function_vect,centriod_vect,estimator_bw_vect, start, end, res=100):
        print('Running analysis grid for all case in list')
        for i,case in enumerate(self.case_list[start:end]):
            print('Running array analysis for case: '+str(start+i))
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
    
    def get_attribute_from_partial_case_list_analysis_array(self,desired_attribute, start, end): # Returns a list of arrays with the requested data
        #for use with run_partial_analysis_array
        temp_data_list=[]
        for i,case in enumerate(self.case_list[start:end]): # Loop through and collect the data
            temp_data_list.append(case.get_attribute_from_array(desired_attribute))
            
        return temp_data_list
    
    def clear_all_analysis(self):
        for i,case in enumerate(self.case_list): # Loop through and collect the data
            case.clear_all_analysis()

class case_obj:
    def __init__(self,sample_curve,true_mob,true_jp,training_data):
        self.sample_curve=sample_curve
        self.true_mob=true_mob
        self.true_jp=true_jp
        self.training_data=training_data # Pass by reference
        self.analysis_list=[]

    def update_scores(self, style = "concat"):
        #style = "concat" for concatenating sub-populations, then calculate l2 norm
        #style = "multiply" for calculating l2 norms for each sub-pop separately, then multiply together
        if style == "concat":
            #self.scores=np.zeros(self.training_data.n)
            #original scoring style: concatenate sub-populations, then calculate l2 norm
            ## self.scores=np.sum((self.training_data.I_ABM_ALL - np.expand_dims(self.sample_curve,axis=[0]))**2,axis=1)**0.5 #Add 2 extra dims to the single curve to make it project
            self.scores=np.sum((self.training_data.I_ABM_ALL - np.expand_dims(self.sample_curve,axis=0))**2,axis=(1,2))**0.5
        if style == "multiply":
            #new scoring style: calculate l2 norms for each sub-pop separately, then multiply together
            # self.scores = np.prod(np.sum((self.training_data.I_ABM_ALL - np.expand_dims(self.sample_curve,axis=0))**2,axis=1)**0.5, axis = 1)
            diff = self.training_data.I_ABM_ALL - np.expand_dims(self.sample_curve, axis=0)
            l2_norms = np.sqrt(np.sum(diff ** 2, axis=1))  # Calculate L2 norm for each time step and each sub-population
            self.scores = np.prod(l2_norms, axis=1)  # Multiply L2 norms for both sub-populations
            # print(self.scores.shape)
        if style == "fraction_diff":
            #new scoring style: concatenate sub-populations, then calculate l2 norm between the differences as a fraction of test data
            # self.scores = np.prod(np.sum((self.training_data.I_ABM_ALL - np.expand_dims(self.sample_curve,axis=0))**2,axis=1)**0.5, axis = 1)
            diff = (self.training_data.I_ABM_ALL - np.expand_dims(self.sample_curve, axis=0))/self.training_data.I_ABM_ALL
            l2_norms = np.sqrt(np.sum(diff ** 2, axis=(1,2)))  # Calculate L2 norm for each time step and each sub-population
            # print(self.scores.shape)


    def make_single_analysis(self,weight_function,centriod,estimator_bw,res=100):
        self.single_res=res
        self.analysis_single=analysis_obj(self,weight_function,centriod,estimator_bw,res)
        self.analysis_list.append(self.analysis_single)

    def make_analysis_array(self,weight_function_vect,centriod_vect,estimator_bw_vect,res=100,verbos=True):
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
            
            # return np.reshape(np.array(temp_data_list),np.shape(self.analysis_array)+np.shape(temp_data_list[1])) # Reshape the data into a grid
            return np.reshape(np.array(temp_data_list),np.shape(self.analysis_array)+np.shape(temp_data_list[0])) # Reshape the data into a grid

        else:
            print('Analysis grid does not have attribute: '+desired_attribute)
            return 0
        
    def clear_all_analysis(self):
        # Clear out references
        attributes_to_clear=['analysis_single','analysis_array','analysis_array_ravel']
        for attribute in attributes_to_clear:
            if hasattr(self,attribute):
                setattr(self,attribute,[])
        
        
class analysis_obj():
    def __init__(self,case,weight_function,centriod,estimator_bw,res=100): # Estimator bw MULTIPLIES THE STD
        self.case=case
        self.weight_function=weight_function
        self.centriod=centriod
        self.estimator_bw=estimator_bw
        self.res=res
        # Do the analysis
        self.update_KDE()
        self.update_analysis_2D()
        self.update_analysis_by_axis()

    def update_KDE(self):
        # Get weights for each score
        self.weights=norm_weights(self.weight_function,self.centriod,self.case.scores)

        # 2D test grid at specified res
        self.mob_vect=np.linspace(self.case.training_data.min_mob,self.case.training_data.max_mob,self.res)
        self.jp_vect=np.linspace(self.case.training_data.min_jp,self.case.training_data.max_jp,self.res+1) # +1 makes sure the sizes are different so that axis dont get mixed up anywhere, can delete once running
        self.mob_vect_mids=mids(self.mob_vect) # grid is at midpoints
        self.jp_vect_mids=mids(self.jp_vect) # grid is at midpoints

        self.mob_grid,self.jp_grid=np.meshgrid(self.mob_vect_mids,self.jp_vect_mids,indexing='ij')

        self.KDE=np.zeros_like(self.mob_grid) # KDE defined at mid points
        
        # Calculate cov matrix, feed this a 2xN matrix
        self.estimator_cov_mat=np.cov(np.vstack((self.case.training_data.mobilities,self.case.training_data.jps)),aweights=self.weights)
        self.cov_mat_inv_scaled=np.linalg.inv(self.estimator_cov_mat*(self.estimator_bw**2))

        # Build KDE by summing the gaussians
        # for weight,mob,jp in tqdm(zip(self.weights,self.case.training_data.mobilities,self.case.training_data.jps)):
        #     if weight>0:
        #         # Return new array of values
        #         new_vals=get_guassian_on_array_2D_cov(self.mob_grid,self.jp_grid,np.array([mob,jp]),self.cov_mat_inv_scaled)
        #         # Normalize
        #         new_vals=new_vals/np.sum(new_vals)*weight
        #         # Add to KDE
        #         self.KDE+=new_vals
        # # Normalize over all KDE
        # self.KDE=self.KDE/np.sum(self.KDE)

        #non-parallelized jit code:
        # start_time = time.time()
        # self.KDE = KDE_calc(self.weights,self.case.training_data.mobilities,self.case.training_data.jps, self.mob_grid, self.jp_grid, self.cov_mat_inv_scaled, self.KDE)
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # # Print the elapsed time
        # print(f"Elapsed Time: {elapsed_time} seconds")

        #parallelized jit code:
        start_time = time.time()
        self.KDE = KDE_calc_parallel(self.weights,self.case.training_data.mobilities,self.case.training_data.jps, self.mob_grid, self.jp_grid, self.cov_mat_inv_scaled, self.KDE)
        end_time = time.time()
        elapsed_time = end_time - start_time
        # print(f"Elapsed Time: {elapsed_time} seconds")

    def update_analysis_2D(self):

        self.inside_95,self.inside_hollow_95=inv_cdf_2D(self.KDE, 0.95)
        self.inside_50,self.inside_hollow_50=inv_cdf_2D(self.KDE, 0.50)

        # interp=sci.interpolate.RegularGridInterpolator((self.mob_vect_mids,self.jp_vect_mids),self.KDE,method='nearest',bounds_error=False,fill_value=None)
        # self.continous_rank=np.sum(np.where(interp((self.case.true_mob,self.case.true_jp)))<=self.KDE)/np.size(self.KDE) #i think this should actually be: self.continous_rank=np.sum(np.where(interp((self.case.true_mob,self.case.true_jp))<=self.KDE))/np.size(self.KDE), but don't need it right now anyway

        #Check how many matches are included (non-zero weights)
        self.non_zero_matches = np.sum(np.where(self.weights>0,1,0))

        #Calculate continuous ranked probability score
        #d_mob = (self.case.training_data.max_mob-self.case.training_data.min_mob)/self.res
        #self.crps = d_mob*np.sum((self.CDF-(self.grid>=self.case.true_mob))**2)

        # Pass?
        interp_95=sci.interpolate.RegularGridInterpolator((self.mob_vect_mids,self.jp_vect_mids), self.inside_95, method='nearest',bounds_error=False,fill_value=None)
        interp_50=sci.interpolate.RegularGridInterpolator((self.mob_vect_mids,self.jp_vect_mids), self.inside_50, method='nearest',bounds_error=False,fill_value=None)

        self.range_pass_95=interp_95((self.case.true_mob,self.case.true_jp))
        self.range_pass_50=interp_50((self.case.true_mob,self.case.true_jp))


    def update_analysis_by_axis(self):

        # Make KDE for each axis
        self.KDE_mob=np.sum(self.KDE,axis=1) # Integrate out jp on axis 1
        self.KDE_jp=np.sum(self.KDE,axis=0)

        # MOB ANALYSIS

        # Make CDF
        self.CDF_mob=np.zeros_like(self.mob_vect)
        self.CDF_mob[1:]=np.cumsum(self.KDE_mob) # Leave the first term as zero
        # print(self.grid)
        self.CDF_interp_mob=sci.interpolate.interp1d(self.mob_vect, self.CDF_mob, kind='linear') # Input is mob, output is probability
        self.inv_CDF_interp_mob=sci.interpolate.interp1d(self.CDF_mob,self.mob_vect, kind='linear') # Input is value 0 to 1

        # Post processing
        self.mean_mob=self.inv_CDF_interp_mob(0.5)
        self.continous_rank_mob=self.CDF_interp_mob(self.case.true_mob)

        #Calculate continuous ranked probability score
        d_mob = (self.case.training_data.max_mob-self.case.training_data.min_mob)/self.res
        self.crps_mob = d_mob*np.sum((self.CDF_mob-(self.mob_vect>=self.case.true_mob))**2)

        # JP ANALYSIS
        
        # Make CDF
        self.CDF_jp=np.zeros_like(self.jp_vect)
        self.CDF_jp[1:]=np.cumsum(self.KDE_jp) # Leave the first term as zero
        # print(self.grid)
        self.CDF_interp_jp=sci.interpolate.interp1d(self.jp_vect, self.CDF_jp, kind='linear') # Input is jp, output is probability
        self.inv_CDF_interp_jp=sci.interpolate.interp1d(self.CDF_jp,self.jp_vect, kind='linear') # Input is value 0 to 1

        # Post processing
        self.mean_jp=self.inv_CDF_interp_jp(0.5)
        self.continous_rank_jp=self.CDF_interp_jp(self.case.true_jp)

        #Calculate continuous ranked probability score
        d_jp = (self.case.training_data.max_jp-self.case.training_data.min_jp)/self.res
        self.crps_jp = d_jp*np.sum((self.CDF_jp-(self.jp_vect>=self.case.true_jp))**2)

    def make_kde_plot(self,title='KDE'):
        plt.figure(dpi = 300, figsize = (4,3))
        plt.xlabel('Mobility')
        plt.ylabel('KDE')
        plt.plot(self.grid_mids,self.KDE/np.diff(self.grid))
        # True
        plt.scatter(self.case.true_mob,0,label='True parameter',color='blue',marker='.',s=600)
        # Matches
        plt.scatter(self.case.training_data.mobilities[self.weights>0],self.case.training_data.mobilities[self.weights>0]*0,label='Accepted matches',color='green',marker='x',alpha=0.3,s=50)
        plt.title(title)
        plt.legend()

    def make_kde_plot_2D(self,plot_matches=False):
        plt.figure(dpi = 300, figsize = (4,3))
        colors = [(1,0.75,0.75,c) for c in np.linspace(0,1,100)]
        cmapsoftred = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=2)
        colors = [(1,0,0,c) for c in np.linspace(0,1,100)]
        cmapred = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=2)

        color_1 = (237/255,248/255,251/255)
        color_2 = (35/255,139/255,69/255)
        colors = [color_1, color_2]
        my_colormap = mcolors.LinearSegmentedColormap.from_list("CustomColormap", colors)

        plt.pcolormesh(self.jp_vect, self.mob_vect, self.KDE,shading='auto',cmap=my_colormap)
        plt.colorbar(label='Probability density')
        plt.pcolormesh(self.jp_vect, self.mob_vect, self.inside_hollow_95,shading='auto',cmap=cmapsoftred) # 95 is the soft red line
        plt.pcolormesh(self.jp_vect, self.mob_vect, self.inside_hollow_50,shading='auto',cmap=cmapred)

        # Use empty lines to get legend
        d = []
        plt.plot(d,d,linewidth=2,color = cmapsoftred(1), label='95% CI')
        plt.plot(d,d,linewidth=2,color = cmapred(1), label='50% CI')

        # Plot matches
        if plot_matches:
            colors = [(1,0,0,c) for c in np.linspace(0,1,100)]
            cmapblue = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=100)
            plt.scatter(self.case.training_data.jps,self.case.training_data.mobilities,marker='.',c=self.weights,cmap=cmapblue, edgecolors='none')
            # plt.scatter(self.case.training_data.jps,self.case.training_data.mobilities,marker='.',c=cmap(norm(self.weights)),label='Matches')

        # Use empty lines to get legend
        plt.scatter(d, d, marker='.',color=(1,0,0), edgecolors='none', label = 'Matches')

        # Plot true
        plt.scatter(self.case.true_jp,self.case.true_mob,marker='*',color='blue',label='True parameter')

        plt.legend(bbox_to_anchor=[1.5, 0.7])
        #plt.title('2D KDE with 50% CI and 95% CI in Red')
        plt.ylabel('Mobility')
        plt.xlabel('Jumping Probability')
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        #plt.savefig('./figs/abc-sample'+str(idx)+'.png',bbox_inches='tight')
        plt.show()

####### Helper functions ############

def get_guassian_on_array(grid,mu,sig): # mu is mean sig is std
    return 1/(sig*(2*np.pi)**0.5)*np.e**(-0.5*((grid-mu)/sig)**2)

def get_guassian_on_array_2D(grid_mob,grid_jp,mu_mob,mu_jp,sig_mob,sig_jp): # mu is mean sig is std
    # see https://en.wikipedia.org/wiki/Gaussian_function, A=1 since it is normalized later
    return np.e**(-(     (grid_mob-mu_mob)**2/(2*sig_mob**2)       +     (grid_jp-mu_jp)**2/(2*sig_jp**2)    ))

@nb.jit
def get_guassian_on_array_2D_cov(grid_mob,grid_jp,mu_vec,cov_mat_inv_scaled): # mu is mean sig is std
    grid_mob_centered=grid_mob-mu_vec[0]
    grid_jp_centered=grid_jp-mu_vec[1]

    # https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    return np.e**(-0.5*(     grid_mob_centered**2*cov_mat_inv_scaled[0,0]      +       2*grid_mob_centered*grid_jp_centered*cov_mat_inv_scaled[1,0]       +       grid_jp_centered**2*cov_mat_inv_scaled[1,1]      ))

@nb.njit
def KDE_calc(weights, mobilities, jps, mob_grid, jp_grid, cov_mat_inv_scaled, KDE):
    for weight,mob,jp in zip(weights,mobilities,jps):
        if weight>0:
            # Return new array of values
            new_vals=get_guassian_on_array_2D_cov(mob_grid,jp_grid,np.array([mob,jp]),cov_mat_inv_scaled)
            # Normalize
            new_vals=new_vals/np.sum(new_vals)*weight
            # Add to KDE
            KDE+=new_vals
    # Normalize over all KDE
    KDE=KDE/np.sum(KDE)
    return KDE

@nb.njit(parallel=True)
def KDE_calc_parallel(weights, mobilities, jps, mob_grid, jp_grid, cov_mat_inv_scaled, KDE):
    num_samples = len(weights)
    
    for i in nb.prange(num_samples):
        weight = weights[i]
        mob = mobilities[i]
        jp = jps[i]

        if weight > 0:
            # Return new array of values
            new_vals = get_guassian_on_array_2D_cov(mob_grid, jp_grid, np.array([mob, jp]), cov_mat_inv_scaled)
            # Normalize
            new_vals = new_vals / np.sum(new_vals) * weight
            # Add to KDE
            for j in nb.prange(KDE.shape[0]):
                for k in nb.prange(KDE.shape[1]):
                    KDE[j, k] += new_vals[j, k]

    # Normalize over all KDE
    total_sum = np.sum(KDE)
    KDE = KDE / total_sum if total_sum > 0 else KDE
    return KDE

def mids(x):
    return (x[1:]+x[:-1])/2

def inv_cdf_2D(Z, target):
    Z_vect=Z.ravel()
    inside_vect=np.zeros_like(Z_vect)
    # Sort
    order=np.argsort(Z_vect)[::-1]
    # Sum
    total=0
    for i, arg in enumerate(order): # Start adding cell by cell
        total+=Z_vect[arg]
        inside_vect[arg]=1 # Mark it
        if total>=target:
            break
    
    # Reshape
    inside = np.reshape(inside_vect.T, Z.shape)
    
    inside_pad=np.pad(inside,(1,1),'constant',constant_values=((1,1),(1,1))) # Add a border of 1s
    inside_hollow=np.copy(inside_pad)
    
    # Loop over interior points
    n_x,n_y=np.shape(Z)
    for ix in range(1,n_x-1+2):
        for iy in range(1,n_y-1+2):
            if np.sum(inside_pad[ix-1:ix+2,iy-1:iy+2])==9:
                inside_hollow[ix,iy]=0
            
    return inside,inside_hollow[1:-1,1:-1] # Cut the border off