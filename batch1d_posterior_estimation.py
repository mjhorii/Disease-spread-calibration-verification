
#========================================
#========= Import packages =========
#========================================

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
from amcmc import ammcmc
import sys
import os
from pathlib import Path

from ABM import SEIR_multiple_pops

#========================================
#========= Read command line inputs =========
#========================================
start_sample_ind = int(sys.argv[1])
end_sample_ind = int(sys.argv[2])
#run posterior estimation over samples with index in range (start_sample_ind, end_sample_ind)

#========================================
#========= Set parameters =========
#========================================

#parameters
m = 1 #number of populations
centers = np.array([[0,0]]) #theta, phi or x, y
spread = np.array([0.1]) #standard deviation of normal distribution
pop = np.array([100]) #population
A_1 = 0.01 #theta or x mobility (azimuth mobility)
A_2 = 0.01 #phi or y mobility (inclination mobility)
R = 1 #radius
d_IU = 0.005
E_0 = np.array([0]) #fraction of initially exposed
I_0 = np.array([0.01]) #fraction of initially infected
S_0 = np.array([0.99]) #fraction of initially susceptible
T_E = 11.6 #time from exposure to infectious
T_E_stdev = 1.9 #standard deviation of exposure time
T_I = 18.49 #incubation time
T_I_stdev = 3.71 #standard deviation of infection time
del_t = 0.1 #time step
verlet_iter = 300 #number of steps between updating verlet list
T = 300
rand_seed = 1
g = None
al = None
jumping_times = np.zeros(int(T/del_t)+1)
jump_prob = 0.5
spherical = None
dist = 'Gamma'

#========================================
#========= Import training data =========
#========================================

import pickle5 as pickle

#85000 Data Points Set:
data_file = open('./Data/Training Data/Calibration method 1/new_I_data_One-Pop-Disc.pickle', "rb")
data = pickle.load(data_file)
data_file.close()

T = 300
del_t = 0.1
time_vec = np.linspace(0,T,int(T/del_t)+1)

#Load in mobility and jumping probability values:
parameter_matrix = pd.read_csv('./Data/Training Data/Calibration method 1/variable_parameter_values_One-Pop-Disc.csv', index_col=0).to_numpy()
mobilities = parameter_matrix[:,0]
jumping_probs = parameter_matrix[:,1]
random_seeds = parameter_matrix[:,2]

#Reshape to sort by mobility and jumping prob:
#based on assumption that data is given in shape (num_of_sims, num_of_time_steps, num_of_subpops)
#and that within the list of simulations, the mobility is the "outer" index (changes more slowly)
#and the jumping probability is the "inner" index (changes more quickly)

unique_mobilities = np.unique(mobilities) 
unique_jumping_probs = np.unique(jumping_probs)

num_of_mobilities = unique_mobilities.shape[0]
num_of_jumping_probs = unique_jumping_probs.shape[0]

num_of_random_seeds_per_param_set = round(data.shape[0]/(num_of_mobilities*num_of_jumping_probs)) #assuming equal number of random seeds run for each param set
num_of_time_steps = data.shape[1]
num_of_subpops = 2 #assumes 2 sub-populations

unique_mobilities = np.unique(mobilities) 
num_of_mobilities = unique_mobilities.shape[0]

data_sorted = np.zeros((num_of_mobilities, num_of_random_seeds_per_param_set, num_of_time_steps))

data_raveled = np.ravel(data, order = 'C')
data = np.reshape(data_raveled, data_sorted.shape)
data = np.swapaxes(data, 0, 2)

## Sum training dataset along time segments
    
num_of_intervals = 5 #number of time intervals to split up for likelihood calculation

time_steps = data.shape[0]
intervals = np.arange(0,time_steps,int(time_steps/num_of_intervals))
intervals[-1] = intervals[-1]+time_steps%num_of_intervals
new_I_per_interval = np.zeros((num_of_intervals,data.shape[1],data.shape[2]))

for i in range(num_of_intervals):
    data_ = data[intervals[i]:intervals[i+1],:,:]
    new_I_per_interval[i,:,:] = np.sum(data_, axis = 0)

# Set up approximate zero-standard deviation pdf functions

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
        

# Generate expanded kernel density estimate PDFs for each time segment and mobility value

#-------------------------PARAMETER DEFINITIONS----------------------

shape_param = 0.5
type_ = "linear"

#--------------------------KDE & MEANS, VARIANCES, STDEVS------------------------------

#generate kernel density estimate PDFs for each time segment and mobility value
means = np.zeros((num_of_intervals, num_of_mobilities))
variances = np.zeros((num_of_intervals, num_of_mobilities))
stdevs = np.zeros((num_of_intervals, num_of_mobilities))
stdevs_approx = np.zeros((num_of_intervals, num_of_mobilities)) #approximate stdev for 0-stdev pdfs based on replacement approximation function

KDE_fns = np.zeros((num_of_intervals, num_of_mobilities))
KDE_fns = KDE_fns.astype('object')

for interval in tqdm(range(num_of_intervals)):
    for mob_ind in range(num_of_mobilities):
        means[interval, mob_ind] = np.mean(new_I_per_interval[interval, :, mob_ind])
        variances[interval, mob_ind] = np.var(new_I_per_interval[interval, :, mob_ind])
        stdevs[interval, mob_ind] = np.std(new_I_per_interval[interval, :, mob_ind])
        stdevs_approx[interval, mob_ind] = np.std(new_I_per_interval[interval, :, mob_ind])
        try: 
            kde = gaussian_kde(new_I_per_interval[interval, :, mob_ind]) #generate KDE -- will fail if all values are 0's
            KDE_fns[interval, mob_ind] = kde.pdf
        except: 
            mean = means[interval, mob_ind]
            KDE_fns[interval, mob_ind], stdev_of_approx_fn = return_zero_stdev_pdf(mean, shape_param, type_)  
            stdevs_approx[interval, mob_ind] = stdev_of_approx_fn
            
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

    interpolated_mean = mean_interp(desired_param) #calculate approximate mean of interpolated pdfs at desired parameter
    interpolated_var = var_interp(desired_param) #calculate approximate variance of interpolated pdfs at desired parameter
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
        interpolated_stdev_approx[interval] = zero_stdev_val
    
    #------------------ reshape arrays: -----------------
    interpolated_mean = np.expand_dims(interpolated_mean, axis = -1)
    interpolated_stdev = np.expand_dims(interpolated_stdev, axis = -1)
    interpolated_stdev_approx = np.expand_dims(interpolated_stdev_approx, axis = -1)
    means = np.expand_dims(means, axis = -1)
    stdevs = np.expand_dims(stdevs, axis = -1)
    stdevs_approx = np.expand_dims(stdevs_approx, axis = -1)
    
    #------------------ calculate transformed coordinates: -----------------
    LHS = (x_mesh - interpolated_mean)/(interpolated_stdev_approx)
    
    x0 = LHS*(stdevs_approx[:,param_below_ind])+means[:,param_below_ind]
    x1 = LHS*(stdevs_approx[:,param_above_ind])+means[:,param_above_ind]
    
    sample_at_x0 = np.zeros_like(x0)
    sample_at_x1 = np.zeros_like(x1)

    #------------------ calculate interpolated pdf samples: -----------------
    for interval in range(num_of_intervals):
        pdf_below = KDE_fns[interval, param_below_ind]
        pdf_above = KDE_fns[interval, param_above_ind]
        sample_at_x0[interval,:] = pdf_below(x0[interval,:])
        sample_at_x1[interval,:] = pdf_above(x1[interval,:])

    alpha = (desired_param-unique_param_values[param_below_ind])/(unique_param_values[param_above_ind]-unique_param_values[param_below_ind])

    pdf = (1-alpha)*(stdevs[:,param_below_ind]/interpolated_stdev)*sample_at_x0+(alpha)*(stdevs[:,param_above_ind]/interpolated_stdev)*sample_at_x1
    
    pdf_fns = np.zeros((num_of_intervals))
    pdf_fns = pdf_fns.astype('object')
    
    #---------- convert interpolated pdf samples into continuous functions via interpolation, compile into array: -------------
    for interval in range(num_of_intervals):
        if renormalize == False:
            pdf_fns[interval] = interpolate.interp1d(x_mesh, pdf[interval, :], axis = -1)
        elif renormalize == True:
            approx_integral = scipy.integrate.trapz(pdf[interval, :], x_mesh)
            pdf_fns[interval] = interpolate.interp1d(x_mesh, pdf[interval, :]/approx_integral, axis = -1)
    
    for i in range(num_of_pdfs_with_stdev_zero): #for each interpolated pdf with stdev = 0:
        interval = np.where(stdev_zero_indexes)[0][i]
        mean_val = interpolated_mean[(interval, 0)] #mean of interpolated pdf with stdev 0
        if renormalize == False:
            pdf_fns[interval], _ = return_zero_stdev_pdf(mean_val, shape_param, type_)
        elif renormalize == True:
            integral = return_integral_of_zero_stdev_pdf(mean_val, shape_param, type_, integral_range = (mesh_min_extended, mesh_max_extended))
            pdf_fns[interval], _ = return_zero_stdev_pdf(mean, shape_param, type_, renormalize = True, integral = integral)
            
    return x_mesh, pdf_fns, interpolated_mean, interpolated_var, np.squeeze(interpolated_stdev_approx)

    
# MCMC
def f_pdf(sample_point, new_I_per_interval_ref):
    sample_point = sample_point[0]
    if sample_point<=0.025 and sample_point>=0.005: #don't like to have this hard-coded here either but also interpolation doesn't work otherwise, and computation time is wasted because prior will be 0 anyway
        x_grid, KDE, interpolated_mean, interpolated_var, interpolated_stdev_approx = interp_KDE_1d(sample_point, unique_mobilities, 
              means, variances, stdevs, stdevs_approx,KDE_fns, 
              mesh_number = 201, mesh_min = 0, mesh_max = 100, 
                  mesh_min_extended = 0, mesh_max_extended = 100, add_to_stdev = 0, shape_param = 0.5, type_ = "linear", renormalize=True)

        log_probs = np.zeros_like(KDE)
        for interval in range(num_of_intervals):
            log_probs[interval] = np.log(KDE[interval](new_I_per_interval_ref[interval]))
        if np.sum(log_probs) == -np.inf:
            return [-1e100,0]
        else:
            return [np.sum(log_probs),0]
    else:
        return [-1e100,0]
    
### Import test/sample data:
RUN_NAME = 'One-Pop'

data_file = open('./Data/Test Data/One-parameter case/new_I_data_One-Pop-NEW-COMBINED-TEST.pickle', "rb")
data = pickle.load(data_file)
data_file.close()

parameter_matrix = pd.read_csv('./Data/Test Data/One-parameter case/variable_parameter_values_One-Pop-NEW-COMBINED-TEST.csv', index_col=0).to_numpy()


#Load in mobility and jumping probability values:
mobilities = parameter_matrix[:,0]
random_seeds = parameter_matrix[:,1]
num_of_samples = mobilities.shape[0]

intervals = np.arange(0,num_of_time_steps,int(num_of_time_steps/num_of_intervals))
intervals[-1] = intervals[-1]+num_of_time_steps%num_of_intervals
new_I_per_interval = np.zeros((num_of_samples, num_of_intervals))

for i in range(num_of_intervals):
    data_ = data[:,intervals[i]:intervals[i+1]]
    new_I_per_interval[:,i] = np.sum(data_, axis = 1)

#========================================
#========= Running Posterior estimation =========
#========================================

print('Running posterior estimation for samples with indexes (',start_sample_ind,",",end_sample_ind)

def posterior_estimation(sample_data_set_ind):

    #------------------Choose Trial dataset:---------------------
    new_I_per_interval_ref = new_I_per_interval[sample_data_set_ind]
    print('actual mobility:', mobilities[sample_data_set_ind])

    #-----------------Run Trial MCMC---------------------

    nsamples = 75000
    nburn = 5000

    seed=100
    nskip = 0
    nthin = 1
    tmpchn_dir = "./Data/MCMC_results/"+RUN_NAME
    logfile_dir = ".Data/MCMC_results/"+RUN_NAME
    os.makedirs(tmpchn_dir, exist_ok = True)
    os.makedirs(logfile_dir, exist_ok = True)

    tmpchn = tmpchn_dir + "/amcmc_TMP_ABM_sample_ind_"+str(sample_data_set_ind)+".dat"
    logfile = logfile_dir + "/amcmc_LOG_ABM_sample_ind_"+str(sample_data_set_ind)+".dat"

    if os.path.isfile(tmpchn): #remove previous run if it exists
        os.remove(tmpchn)
    if os.path.isfile(logfile): #remove previous run if it exists
        os.remove(logfile)

    opts = {"nsteps": nsamples, "nfinal": 10000000,"gamma": 1,
            "inicov": np.array([0.001]),"inistate": np.array([0.0151]),
            "spllo": np.array([0.005]),"splhi": np.array([0.025]),
            "logfile": logfile,"burnsc":5,
            "nburn":nburn,"nadapt":100,"coveps":1.e-10,"ofreq":5000,"tmpchn":tmpchn,'rnseed':sample_data_set_ind
            }

    ndim = 1
    np.random.seed(seed)

    print('Sampling f_pdf function with AMCMC ...')
    sol=ammcmc(opts,f_pdf,new_I_per_interval_ref)
    samples = sol['chain']
    logprob = sol['minfo'][:,1]

    import matplotlib.pyplot as plt
    samples = samples[nskip::nthin]
    logprob = logprob[nskip::nthin]

    #save data:
    with open(tmpchn_dir+'/AMCMC_sample_ind_'+str(int(sample_data_set_ind))+'.pickle', 'wb') as handle:
        pickle.dump(sol, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Acceptance rate',sol['accr'])
    print('Mean:',np.mean(samples, axis=0))
    print('Var:',np.var(samples, axis=0))
    print('Cov:',np.cov(samples.T)) 

    #----------Brute force sampling----------

    save_dir = './Data/Brute_force_posterior_estimation/'+RUN_NAME
    os.makedirs(save_dir, exist_ok = True)
    save_file = save_dir+'/IND'+str(int(sample_data_set_ind))+'LOG_PROBS.txt'

    if os.path.isfile(save_file): #remove previous run if it exists
        os.remove(save_file)

    n_m = 200
    test_mobilities = np.linspace(unique_mobilities[0]+1E-10, unique_mobilities[-1]-1E-10, n_m)

    log_probs = np.zeros((test_mobilities.shape[0]))

    for j in range(test_mobilities.shape[0]):
        log_probs[j] = f_pdf([test_mobilities[j]], new_I_per_interval_ref)[0]

        fout = open(save_file, 'ab')
        dataout = np.array([[test_mobilities[j],log_probs[j]]])
        np.savetxt(fout, dataout, fmt='%.8e',delimiter=' ', newline='\n')
        fout.close()

    with open(save_dir+'/IND'+str(int(sample_data_set_ind))+'LOG_PROBS.pickle', 'wb') as handle:
        pickle.dump(log_probs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

from multiprocessing import Pool
import multiprocessing as mp
import time
import math

for i in range(start_sample_ind,end_sample_ind):
    posterior_estimation(i)

# mp.set_start_method('forkserver', force=True)

# num_workers = mp.cpu_count()  
# # num_workers = 25
# print(num_workers)
# start_time = time.perf_counter()
# with Pool(num_workers) as pool:
#     result = pool.map(posterior_estimation, range(start_sample_ind,end_sample_ind))
# finish_time = time.perf_counter()
# print("Program finished in {} seconds - using multiprocessing".format(finish_time-start_time))
# print("---")