from __future__ import division
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import math
import time
from tqdm import tqdm
import cProfile
import pickle
from scipy.stats import gaussian_kde
from scipy import interpolate
import pandas as pd
from math import sqrt
import scipy.stats as stats
from matplotlib.pyplot import imshow
import sys
import os
from pathlib import Path

from ABM import SEIR_multiple_pops
from amcmc import ammcmc

#========================================
#========= Read command line inputs =========
#========================================
start_sample_ind = int(sys.argv[1])
end_sample_ind = int(sys.argv[2])
#run posterior estimation over samples with index in range (start_sample_ind, end_sample_ind

#========================================
#========= Import training data =========
#========================================

data_file = open('./Data/Training Data/Two-parameter case/Calibration method 1/new_I_data_Two-Pop-Disc.pickle', "rb")
data_sorted = pickle.load(data_file)
data_file.close()

#Change data type to save memory:
print('Size before changing data type: ', data_sorted.nbytes)

if np.max(data_sorted) < 256:
    data_sorted = data_sorted.astype('uint8')
elif np.max(data_sorted) < 65536:
    data_sorted = data_sorted.astype('uint16')
    
print('Size after changing data type: ', data_sorted.nbytes)

#Load in mobility and jumping probability values:
# load parameter matrix from previous dataset just to get unique mobility and jumping prob values
parameter_matrix = pd.read_csv('./Data/Training Data/Two-parameter case/Calibration method 1/variable_parameter_values_Two-Pop-Discrete-Sample.csv', index_col=0).to_numpy()
mobilities = parameter_matrix[:,0]
jumping_probs = parameter_matrix[:,1]
random_seeds = parameter_matrix[:,2]

unique_mobilities = np.unique(mobilities) 
unique_jumping_probs = np.unique(jumping_probs)

num_of_mobilities = unique_mobilities.shape[0]
num_of_jumping_probs = unique_jumping_probs.shape[0]

num_of_random_seeds_per_param_set = 5000
num_of_time_steps = 3001
num_of_subpops = 2 #assumes 2 sub-populations


#Load in constant parameter values, use to create time vector:
# again, loading from previous dataset
const_parameter_matrix = pd.read_csv('./Data/Training Data/Two-parameter case/Calibration method 1/constant_parameters_values_Two-Pop-Discrete-Sample.csv', index_col=0)
T = const_parameter_matrix['Total time (T)'][0]
del_t = const_parameter_matrix['Time step (del_t)'][0]
time_vec = np.linspace(0,T,int(T/del_t)+1)

num_of_intervals = 5 #number of time intervals to split up for likelihood calculation


intervals = np.arange(0,num_of_time_steps,int(num_of_time_steps/num_of_intervals))
intervals[-1] = intervals[-1]+num_of_time_steps%num_of_intervals
new_I_per_interval = np.zeros((num_of_intervals, num_of_mobilities, num_of_jumping_probs, num_of_random_seeds_per_param_set, num_of_subpops))

for i in range(num_of_intervals):
    data_ = data_sorted[:,:,:,intervals[i]:intervals[i+1],:]
    new_I_per_interval[i,:,:,:,:] = np.sum(data_, axis = 3)


def return_zero_stdev_pdf(mean, shape_param, type_, renormalize = False, integral = 1):
    """
    function to return a function representing/approximating a pdf with zero standard deviation
        mean:        value at which pdf/probability is non-zero
        shape_param: if type_ is "gaussian", represents standard deviation. if type_ is "linear", represents the distance away from the mean at which probability becomes zero 
                     (e.g., if mean is 0, shape_param is 1, pdf is non-zero between -1 and 1).
        type_:       if "gaussian," approximates pdf shape with gaussian curve (mean = mean, stdev = shape_param). if "linear," approximates pdf shape with a piecewise linear function
                     where if value is greater than mean+shape_param or less than mean-shape_param, value is 0. otherwise, it is a linear function with maximum at the mean.
                     the resulting function should be a valid pdf (as in, area under curve sums to one)
        renormalize: if True, divide by the integral value to normalize. if False, do nothing
        integral:    value to normalize by

    """
    
    if type_ == "gaussian":
        def zero_stdev_pdf_gaussian(x):
            return scipy.stats.norm.pdf(x,mean,shape_param)
        stdev_of_gaussian = shape_param
        return zero_stdev_pdf_gaussian, stdev_of_gaussian
    
    if type_ == "linear":
        if renormalize == False:
            def zero_stdev_pdf_linear(x):
                slope = 1/shape_param
                result = np.zeros_like(x)
                try: #for non-scalar inputs:
                    for i in range(x.shape[0]):
                        if x[i] > mean + shape_param:
                            result[i] = 0
                        elif x[i] < mean - shape_param:
                            result[i] = 0                 
                        elif x[i] > mean:
                            result[i] = 1 - (x[i] - mean)*slope
                        elif x[i] <= mean:
                            result[i] = 1 - (mean - x[i])*slope
                except: #for scalar inputs:
                    if np.isscalar(x):
                        if x > mean + shape_param:
                            result = 0
                        elif x < mean - shape_param:
                            result = 0                 
                        elif x > mean:
                            result = 1 - (x - mean)*slope
                        elif x <= mean:
                            result = 1 - (mean - x)*slope
                return result/(1*shape_param) #normalized so that resulting pdf area sums to one

        elif renormalize == True:
            def zero_stdev_pdf_linear(x):
                slope = 1/shape_param
                result = np.zeros_like(x)
                try: #for non-scalar inputs:
                    for i in range(x.shape[0]):
                        if x[i] > mean + shape_param:
                            result[i] = 0
                        elif x[i] < mean - shape_param:
                            result[i] = 0                 
                        elif x[i] > mean:
                            result[i] = 1 - (x[i] - mean)*slope
                        elif x[i] <= mean:
                            result[i] = 1 - (mean - x[i])*slope
                except: #for scalar inputs:
                    if np.isscalar(x):
                        if x > mean + shape_param:
                            result = 0
                        elif x < mean - shape_param:
                            result = 0                 
                        elif x > mean:
                            result = 1 - (x - mean)*slope
                        elif x <= mean:
                            result = 1 - (mean - x)*slope
                return (result/(1*shape_param))/integral #normalized so that resulting pdf area sums to one
            
        stdev_of_linear_fn = sqrt(shape_param**2/6)
        return zero_stdev_pdf_linear, stdev_of_linear_fn

    
def return_integral_of_zero_stdev_pdf(mean, shape_param, type_, integral_range):
    """
    returns the analytically-calculated integral of the zero-stdev approximation pdf function between specified numbers in integral_range
    """
    if type_ == "linear":
        total_area = 1
        range_max = integral_range[1]
        range_min = integral_range[0]
        if range_max < range_min:
            print('error: integral_range max < integral_range min')
            return
        if range_min > mean - shape_param:
            f_at_range_min = 1/shape_param - np.abs(mean - range_min)/(shape_param**2)
            if mean > range_min:
                total_area += -f_at_range_min*(shape_param - np.abs(mean - range_min))/2
            elif mean <= range_min:
                total_area = f_at_range_min*(shape_param - np.abs(mean - range_min))/2
                
        if range_max < mean + shape_param:
            f_at_range_max = 1/shape_param - np.abs(mean - range_max)/(shape_param**2)
            if mean > range_max:
                area_to_remove = 1 - f_at_range_max*(shape_param - np.abs(mean - range_max))/2
                total_area += -area_to_remove
            elif mean <= range_max:
                total_area += -f_at_range_max*(shape_param - np.abs(mean - range_max))/2
            
        return total_area
    else:
        print('only computes integral for linear case currently')
        

#-------------------------PARAMETER DEFINITIONS----------------------

shape_param = 0.4
type_ = "linear"

#--------------------------NON-EXAPNDED KDE & MEANS, VARIANCES, STDEVS------------------------------

#generate kernel density estimate PDFs for each time segment and mobility value
means = np.zeros((num_of_intervals, num_of_mobilities, num_of_jumping_probs, num_of_subpops))
variances = np.zeros((num_of_intervals, num_of_mobilities, num_of_jumping_probs, num_of_subpops))
stdevs = np.zeros((num_of_intervals, num_of_mobilities, num_of_jumping_probs, num_of_subpops))
stdevs_approx = np.zeros((num_of_intervals, num_of_mobilities, num_of_jumping_probs, num_of_subpops)) #approximate stdev for 0-stdev pdfs based on replacement approximation function

KDE_fns = np.zeros((num_of_intervals, num_of_mobilities, num_of_jumping_probs, num_of_subpops))
KDE_fns = KDE_fns.astype('object')

for interval in tqdm(range(num_of_intervals)):
    for mob_ind in range(num_of_mobilities):
        for jump_ind in range(num_of_jumping_probs):
            for subpop_ind in range(num_of_subpops):
                means[interval, mob_ind, jump_ind, subpop_ind] = np.mean(new_I_per_interval[interval, mob_ind, jump_ind, :, subpop_ind])
                variances[interval, mob_ind, jump_ind, subpop_ind] = np.var(new_I_per_interval[interval, mob_ind, jump_ind, :, subpop_ind])
                stdevs[interval, mob_ind, jump_ind, subpop_ind] = np.std(new_I_per_interval[interval, mob_ind, jump_ind, :, subpop_ind])
                stdevs_approx[interval, mob_ind, jump_ind, subpop_ind] = np.std(new_I_per_interval[interval, mob_ind, jump_ind, :, subpop_ind])
                try: 
                    kde = gaussian_kde(new_I_per_interval[interval, mob_ind, jump_ind, :, subpop_ind]) #generate KDE -- will fail if all values are 0's
                    KDE_fns[interval, mob_ind, jump_ind, subpop_ind] = kde.pdf
                except: 
                    mean = means[interval, mob_ind, jump_ind, subpop_ind]
                    KDE_fns[interval, mob_ind, jump_ind, subpop_ind], stdev_of_approx_fn = return_zero_stdev_pdf(mean, shape_param, type_)  
                    stdevs_approx[interval, mob_ind, jump_ind, subpop_ind] = stdev_of_approx_fn

def power(x, k):
    x_new = np.abs(x)**k
    x_new = x_new*np.sign(x)
    return x_new

def interp_KDE_1d(desired_param, unique_param_values, means, variances, stdevs, stdevs_approx,
                  KDE_fns, mesh_number, mesh_min, mesh_max, 
                  mesh_min_extended, mesh_max_extended, add_to_stdev, shape_param, type_, renormalize=False):
    """
    function to interpolate between two PDFs
    
    desired_param = value of parameter to interpolate PDF at
    unique_param_values = parameter values at which there exists a known PDF
    renormalize = "True" or "False". if True, will crop final PDF between values in x_mesh and renormalize the PDF to have an integral of one over this new range.
    """
    #means and variances inputs must have the param you wish to interpolate over at the last axis
    
    #------------------ create mesh of points to sample from interpolated pdf: -----------------
    x_mesh = np.linspace(mesh_min,mesh_max,mesh_number)
    x_mesh[0] = mesh_min_extended
    x_mesh[-1] = mesh_max_extended
    
    #------------------ calculate means and variances of interpolated pdfs: -----------------
    mean_interp = interpolate.interp1d(unique_param_values, means, axis = -1) #create interpolation function for pdf means
    var_interp = interpolate.interp1d(unique_param_values, variances, axis = -1) #create interpolation function for pdf variances

    interpolated_mean = mean_interp(desired_param) #calculate mean of interpolated pdfs at desired parameter
    interpolated_var = var_interp(desired_param) #calculate variance of interpolated pdfs at desired parameter
    interpolated_var[np.where(interpolated_var<0)] = 0 #bug in scipy interp1d can cause negative values -- correcting for it here, otherwise sqrt below fails

    interpolated_stdev = np.sqrt(interpolated_var)
    interpolated_stdev_approx = np.copy(interpolated_stdev) #same as interpolated_stdev, but later replace 0 stdev values with approximate based on zero-stdev pdf function shape
    
    param_below_ind = np.where(unique_param_values <= desired_param)[0][-1] #get the closest parameter grid value below the desired param value
    param_above_ind = int(param_below_ind+1) #get the closest parameter grid value above the desired param value
    
    #------------------ find interpolated pdfs with 0-stdevs, replace interpolated_stdev_approx value: -----------------
    stdev_zero_indexes = (interpolated_stdev == 0)
    num_of_pdfs_with_stdev_zero = np.where(stdev_zero_indexes == True)[0].shape[0] #after interpolation, how many of the interpolated pdfs have stdev 0?
    temp_mean_val = 0
    _, zero_stdev_val = return_zero_stdev_pdf(temp_mean_val, shape_param, type_)
    for i in range(num_of_pdfs_with_stdev_zero): #for each interpolated pdf with stdev = 0:
        interval = np.where(stdev_zero_indexes)[0][i]
        subpop_ind = np.where(stdev_zero_indexes)[1][i]
        interpolated_stdev_approx[interval, subpop_ind] = zero_stdev_val
    
    #------------------ reshape arrays: -----------------
    interpolated_mean = np.expand_dims(interpolated_mean, axis = -1)
    interpolated_stdev = np.expand_dims(interpolated_stdev, axis = -1)
    interpolated_stdev_approx = np.expand_dims(interpolated_stdev_approx, axis = -1)
    means = np.expand_dims(means, axis = -1)
    stdevs = np.expand_dims(stdevs, axis = -1)
    stdevs_approx = np.expand_dims(stdevs_approx, axis = -1)
    
    #------------------ calculate transformed coordinates: -----------------
    LHS = (x_mesh - interpolated_mean)/(interpolated_stdev_approx)
    
    x0 = LHS*(stdevs_approx[:,:,param_below_ind])+means[:,:,param_below_ind]
    x1 = LHS*(stdevs_approx[:,:,param_above_ind])+means[:,:,param_above_ind]
    
    sample_at_x0 = np.zeros_like(x0)
    sample_at_x1 = np.zeros_like(x1)

    #------------------ calculate interpolated pdf samples: -----------------
    for interval in range(num_of_intervals):
        for subpop_ind in range(num_of_subpops):
            pdf_below = KDE_fns[interval, param_below_ind, subpop_ind]
            pdf_above = KDE_fns[interval, param_above_ind, subpop_ind]
            sample_at_x0[interval,subpop_ind,:] = pdf_below(x0[interval,subpop_ind,:])
            sample_at_x1[interval,subpop_ind,:] = pdf_above(x1[interval,subpop_ind,:])

    alpha = (desired_param-unique_param_values[param_below_ind])/(unique_param_values[param_above_ind]-unique_param_values[param_below_ind])

    pdf = (1-alpha)*(stdevs[:,:,param_below_ind]/interpolated_stdev)*sample_at_x0+(alpha)*(stdevs[:,:,param_above_ind]/interpolated_stdev)*sample_at_x1 #using interpolated_stdev instead of interpolated_stdev_approx is ok here because we replace PDFs with stdev = 0 later anyway
    
    pdf_fns = np.zeros((num_of_intervals, num_of_subpops))
    pdf_fns = pdf_fns.astype('object')
    
    #---------- convert interpolated pdf samples into continuous functions via interpolation, compile into array: -------------
    for interval in range(num_of_intervals):
        for subpop_ind in range(num_of_subpops):
            if renormalize == False:
                pdf_fns[interval, subpop_ind] = interpolate.interp1d(x_mesh, pdf[interval, subpop_ind, :], axis = -1)
            elif renormalize == True:
                approx_integral = scipy.integrate.trapz(pdf[interval, subpop_ind, :], x_mesh)
                pdf_fns[interval, subpop_ind] = interpolate.interp1d(x_mesh, pdf[interval, subpop_ind, :]/approx_integral, axis = -1)
                
        #------------------ reevaluate means and variances: -----------------
    
    reeval_x_mesh = power(np.linspace(np.sign(mesh_min)*(abs(mesh_min)**(1/1.5)),np.sign(mesh_max)*(abs(mesh_max)**(1/1.5)),mesh_number),1.5)
    reeval_interpolated_var = np.zeros_like(interpolated_var)
    reeval_interpolated_mean = np.zeros_like(interpolated_mean)
    reeval_interpolated_stdev_approx = np.zeros_like(interpolated_stdev_approx)
    for interval in range(num_of_intervals):
        for subpop_ind in range(num_of_subpops):
            mu = scipy.trapz(pdf_fns[interval, subpop_ind](reeval_x_mesh)*reeval_x_mesh, x = reeval_x_mesh)
            S = scipy.trapz(pdf_fns[interval, subpop_ind](reeval_x_mesh)*(reeval_x_mesh-mu)**2, x = reeval_x_mesh)
            
            reeval_interpolated_var[interval, subpop_ind] = S
            reeval_interpolated_mean[interval, subpop_ind] = mu
            reeval_interpolated_stdev_approx[interval, subpop_ind] = sqrt(S)
    
    for i in range(num_of_pdfs_with_stdev_zero): #for each interpolated pdf with stdev = 0:
        interval = np.where(stdev_zero_indexes)[0][i]
        subpop_ind = np.where(stdev_zero_indexes)[1][i]
        mean_val = interpolated_mean[(interval, subpop_ind, 0)] #mean of interpolated pdf with stdev 0
        reeval_interpolated_mean[interval, subpop_ind] = mean_val
        reeval_interpolated_var[interval, subpop_ind] = (interpolated_stdev_approx[interval, subpop_ind])**2
        reeval_interpolated_stdev_approx[interval, subpop_ind] = interpolated_stdev_approx[interval, subpop_ind] #will be whatever the stdev approx value is
        if renormalize == False:
            pdf_fns[interval, subpop_ind], _ = return_zero_stdev_pdf(mean_val, shape_param, type_)
        elif renormalize == True:
            integral = return_integral_of_zero_stdev_pdf(mean_val, shape_param, type_, integral_range = (mesh_min_extended, mesh_max_extended))
            pdf_fns[interval, subpop_ind], _ = return_zero_stdev_pdf(mean, shape_param, type_, renormalize = True, integral = integral)

    return x_mesh, pdf_fns, reeval_interpolated_mean, reeval_interpolated_var, np.squeeze(reeval_interpolated_stdev_approx)

def interp_KDE_2d(desired_jump_prob, desired_mobility):

    jump_prob_below_ind = np.where(unique_jumping_probs <= desired_jump_prob)[0][-1]
    jump_prob_above_ind = int(jump_prob_below_ind+1)
    
    #------------ Interpolate between mobility values at neighboring jumping probability values: ------------
    
#     print('RUNNING INTERPOLATION BETWEEN MOBILITY AT JUMPING PROBABILITY BELOW \n\n')
    x_mesh, pdf_jump_prob_below, mean_jump_prob_below, var_jump_prob_below, stdevs_approx_below = interp_KDE_1d(desired_mobility, unique_mobilities, 
                                                                                           np.swapaxes(means[:,:,jump_prob_below_ind,:],2,1), 
                                                                                           np.swapaxes(variances[:,:,jump_prob_below_ind,:],2,1), 
                                                                                           np.swapaxes(stdevs[:,:,jump_prob_below_ind,:],2,1),
                                                                                           np.swapaxes(stdevs_approx[:,:,jump_prob_below_ind,:],2,1),
                                                                                           KDE_fns[:,:,jump_prob_below_ind,:],
                                                                                           501, -50, 150, -10000000, 10000000, 0.5, 0.4, "linear",renormalize=True)

#     print('RUNNING INTERPOLATION BETWEEN MOBILITY AT JUMPING PROBABILITY ABOVE \n\n')
    x_mesh, pdf_jump_prob_above, mean_jump_prob_above, var_jump_prob_above, stdevs_approx_above = interp_KDE_1d(desired_mobility, unique_mobilities, 
                                                                                           np.swapaxes(means[:,:,jump_prob_above_ind,:],2,1),
                                                                                           np.swapaxes(variances[:,:,jump_prob_above_ind,:],2,1), 
                                                                                           np.swapaxes(stdevs[:,:,jump_prob_above_ind,:],2,1), 
                                                                                           np.swapaxes(stdevs_approx[:,:,jump_prob_above_ind,:],2,1),
                                                                                           KDE_fns[:,:,jump_prob_above_ind,:], 
                                                                                           501, -50, 150, -10000000, 10000000, 0.5, 0.4, "linear", renormalize=True)

    #-------- Compile PDF, means, variance, and stdev info for jump prob above and below ----------
    KDEs_expanded_interpolated_by_mob = np.array([pdf_jump_prob_below, pdf_jump_prob_above])
    KDEs_expanded_interpolated_by_mob = np.swapaxes(KDEs_expanded_interpolated_by_mob, 0, 1)

    means_interpolated_by_mobility = np.zeros((num_of_intervals, num_of_subpops, 2))
    means_interpolated_by_mobility[:,:,0] = mean_jump_prob_below[:,:,0]
    means_interpolated_by_mobility[:,:,1] = mean_jump_prob_above[:,:,0]

    vars_interpolated_by_mobility = np.zeros((num_of_intervals, num_of_subpops, 2))
    vars_interpolated_by_mobility[:,:,0] = var_jump_prob_below
    vars_interpolated_by_mobility[:,:,1] = var_jump_prob_above
    stdevs_interpolated_by_mobility = np.sqrt(vars_interpolated_by_mobility)
    
    stdevs_interpolated_by_mobility_approx = np.zeros((num_of_intervals, num_of_subpops, 2))
    stdevs_interpolated_by_mobility_approx[:,:,0] = stdevs_approx_below
    stdevs_interpolated_by_mobility_approx[:,:,1] = stdevs_approx_above
    
    #-------------- Interpolate between jumping probability values -----------------
    x_mesh_interp, pdf, mean_, var_, stdevs_approx_ = interp_KDE_1d(desired_jump_prob, np.array([unique_jumping_probs[jump_prob_below_ind], 
                                unique_jumping_probs[jump_prob_above_ind]]), means_interpolated_by_mobility, 
                                vars_interpolated_by_mobility, stdevs_interpolated_by_mobility, stdevs_interpolated_by_mobility_approx,
                                KDEs_expanded_interpolated_by_mob, 201, 0, 100, 0, 100, 0.5, 0.5, "linear", renormalize = True)

    return pdf


def sample_from_mult_pdf(pdfs, num, x_grid):
    """
    inverse transform sampling of multiple pdfs
    pdfs: value of pdfs along x_grid values with shape (x_grid dimension, num of pdfs)
    num: number of random values to sample from each pdf
    x_grid: grid of values along which values of pdf are given
    """
    cdf = np.cumsum(pdfs, axis = 0)
    cdf = cdf / cdf[-1,:]
    values = np.random.rand(num, pdfs.shape[1], pdfs.shape[2])
    random_from_cdf = np.zeros((num, pdfs.shape[1], pdfs.shape[2]))
    for i in range(pdfs.shape[1]):
        for j in range(pdfs.shape[2]):
            value_bins = np.searchsorted(cdf[:,i,j], values[:,i,j]) #would be better if this could be applied all at once, but np.searchsorted doesn't have an option to specify axis or anything
            random_from_cdf[:,i,j] = x_grid[value_bins]
    return random_from_cdf
    
def f_pdf(sample_point, new_I_per_interval_ref):
    """
    function that takes a sample_point [sample_mobility, sample_jump_prob] and returns posterior log-likelihood given data in new_I_per_interval_ref
    
    posterior likelihood is calculated by first calculating interpolated PDFs at the sample_point parameter values.
    then, the log-likelihood values are simply sampled directly from these PDFs based on the given data.
    """
    
    sample_mobility = sample_point[0]
    sample_jump_prob = sample_point[1]
    
    if sample_mobility<=0.025 and sample_mobility>=0.005: #don't like to have this hard-coded here either but also interpolation doesn't work otherwise, and computation time is wasted because prior will be 0 anyway
        if sample_jump_prob<=0.1 and sample_jump_prob>=0:
            KDE = interp_KDE_2d(sample_jump_prob, sample_mobility)
            log_probs = np.zeros_like(KDE)
            
            for interval in range(num_of_intervals):
                for subpop_ind in range(num_of_subpops):
                    log_probs[interval, subpop_ind] = np.log(KDE[interval, subpop_ind](new_I_per_interval_ref[interval, subpop_ind]))

        if np.sum(log_probs) == -np.inf:
            return [-1e100,0]
        else:
            return [np.sum(log_probs),0]
    else:
        return [-1e100,0]
    
#set up a sampling distribution
#gaussian centered at current observation point:
def sample(x, scale): #scale is variance
    scale = math.sqrt(scale)
    return scipy.stats.norm.rvs(loc=x, scale=scale, size=1)[0]

#Metropolis Hastings Algorithm
def MH_A(num_iterations,x0,n0,epsilon,s_d,init_scale,f,sample_fn,prior=None):
    np.random.seed(1)
    sequence = []
    x = x0
    accepted = 1
    sequence.append(x)
    scale = init_scale
    iter_since_last_accepted = 0
    same_as_prev = False
    for i in tqdm(range(num_iterations)):
        if same_as_prev == False:
            p_x = f(x)
        if prior != None:
            p_x = p_x*prior(x)
        
        if i > n0:
            scale = s_d*np.var(sequence)+s_d*epsilon #should replace with recursive formula
            
        x_new = sample(x, scale)
        p_x_new = f(x_new)
        if prior != None:
            p_x_new = p_x_new*prior(x_new)
        
        ratio = p_x_new/p_x

        u = np.random.rand(1)

        if u <= ratio:
            x = x_new
            accepted += 1
            same_as_prev = True
            p_x = p_x_new
            iter_since_last_accepted = 0
        elif iter_since_last_accepted > 10: #if it seems like it's stuck, make it reevaluate the current parameter
#             p_x = f(x)
#             if prior != None:
#                 p_x = p_x*prior(x)
            same_as_prev = False
            iter_since_last_accepted = 0 
        else:
            same_as_prev = True
            iter_since_last_accepted +=1
        
        sequence.append(x)
        
    print('Fraction accepted:', accepted/num_iterations)
        
    return sequence

def prior(x):
    if x > 0.0025:
        return 0
    elif x < 0.0005:
        return 0
    else:
        return 1


#LOAD TEST DATA:
RUN_NAME = 'Two-Pop'

data_file = open('./Data/Test Data/Two-parameter case/new_I_data_Two-Pop-NEW-COMBINED-TEST.pickle', "rb")
data = pickle.load(data_file)
data_file.close()

#Load in mobility and jumping probability values:
parameter_matrix = pd.read_csv('./Data/Test Data/Two-parameter case/variable_parameter_values_Two-Pop-NEW-COMBINED-TEST.csv', index_col=0).to_numpy()
mobilities = parameter_matrix[:,0]
jumping_probs = parameter_matrix[:,1]
random_seeds = parameter_matrix[:,2]
num_of_samples = mobilities.shape[0]

intervals = np.arange(0,num_of_time_steps,int(num_of_time_steps/num_of_intervals))
intervals[-1] = intervals[-1]+num_of_time_steps%num_of_intervals
new_I_per_interval = np.zeros((num_of_samples, num_of_intervals, num_of_subpops))

for i in range(num_of_intervals):
    data_ = data[:,intervals[i]:intervals[i+1],:]
    print(data_.shape)
    new_I_per_interval[:,i,:] = np.sum(data_, axis = 1)
    
def posterior_estimation(sample_data_set_ind):
    
    new_I_per_interval_ref = new_I_per_interval[sample_data_set_ind]
    
    #-------- MCMC ----------
    nsamples = 75000
    seed=100
    nburn = 5000

    nskip = 0
    nthin = 1
    tmpchn_dir = './Data/MCMC_results/'+RUN_NAME
    logfile_dir = './Data/MCMC_results/'+RUN_NAME
    
    os.makedirs(tmpchn_dir, exist_ok = True)
    os.makedirs(logfile_dir, exist_ok = True)

    tmpchn = "./Data/MCMC_results/"+RUN_NAME+"/amcmc_TMP_ABM_sample_ind_"+str(sample_data_set_ind)+".dat"
    logfile = "./Data/MCMC_results/"+RUN_NAME+"/amcmc_LOG_ABM_sample_ind_"+str(sample_data_set_ind)+".dat"
    if os.path.isfile(tmpchn): #remove previous run if it exists
        os.remove(tmpchn)
    if os.path.isfile(logfile): #remove previous run if it exists
        os.remove(logfile)
    opts = {"nsteps": nsamples, "nfinal": 10000000,"gamma": 1,
            "inicov": np.array([[1E-3,0],[0,1E-3]]),"inistate": np.array([0.015, 0.0005]),
            "spllo": np.array([0.005,0]),"splhi": np.array([[0.025,0.001]]),
            "logfile": logfile,"burnsc":5,
            "nburn":nburn,"nadapt":100,"coveps":1.e-10,"ofreq":50,"tmpchn":tmpchn,'rnseed':int(sample_data_set_ind)
            }

    ndim = 2
    np.random.seed(seed)
    theta0 = np.array([0.0151, 0.00051])
    print(f'Start point:{theta0}')
    opts["inistate"] = theta0

    print('Sampling likelihood function with AMCMC ...')
    sol=ammcmc(opts,f_pdf,new_I_per_interval_ref)
    samples = sol['chain']
    logprob = sol['minfo'][:,1]

    import matplotlib.pyplot as plt
    samples = samples[nskip::nthin]
    logprob = logprob[nskip::nthin]

    print('Acceptance rate',sol['accr'])
    print('Mean:',np.mean(samples, axis=0))
    print('Var:',np.var(samples, axis=0))
    print('Cov:',np.cov(samples.T)) 

    #save data:
    with open('./Data/MCMC_results/'+RUN_NAME+'/AMCMC_sample_ind_'+str(int(sample_data_set_ind))+'.pickle', 'wb') as handle:
        pickle.dump(sol, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #----------Brute force sampling----------
    save_dir = './Data/Brute_force_posterior_estimation/'+RUN_NAME
    os.makedirs(save_dir, exist_ok = True)
    save_file = save_dir+'/IND'+str(int(sample_data_set_ind))+'LOG_PROBS.txt'
    
    if os.path.isfile(save_file): #remove previous run if it exists
        os.remove(save_file)

    n_j = 75
    n_m = 76
    test_jump_probs = np.linspace(unique_jumping_probs[0]+1E-10, unique_jumping_probs[-1]-1E-10, n_j)
    test_mobilities = np.linspace(unique_mobilities[0]+1E-10, unique_mobilities[-1]-1E-10, n_m)

    log_probs = np.zeros((test_jump_probs.shape[0], test_mobilities.shape[0]))
    tmparg = 0

    for i in tqdm(range(test_jump_probs.shape[0])):
        for j in range(test_mobilities.shape[0]):
            log_probs[i,j] = f_pdf([test_mobilities[j], test_jump_probs[i]], new_I_per_interval_ref)[0]
        
            fout = open(save_file, 'ab')
            dataout = np.array([[test_mobilities[j],test_jump_probs[i],log_probs[i,j]]])
            np.savetxt(fout, dataout, fmt='%.8e',delimiter=' ', newline='\n')
            fout.close()

    with open(save_dir+'/IND'+str(int(sample_data_set_ind))+'LOG_PROBS.pickle', 'wb') as handle:
        pickle.dump(log_probs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
from multiprocessing import Pool
import multiprocessing as mp
import time
import math

# mp.set_start_method('forkserver', force=True)

for i in range(start_sample_ind,end_sample_ind):
    posterior_estimation(i)

# num_workers = mp.cpu_count()  
# # num_workers = 25
# print('Number of workers:', num_workers)
# start_time = time.perf_counter()
# with Pool(num_workers) as pool:
#     result = pool.map(posterior_estimation, range(start_sample_ind,end_sample_ind))
#     # result = pool.map(posterior_estimation, indicies_to_run)
# finish_time = time.perf_counter()
# print("Program finished in {} seconds - using multiprocessing".format(finish_time-start_time))
# print("---")