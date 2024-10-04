import numpy as np
from scipy import interpolate
from math import sqrt
import scipy

# ------------------------------------------------------------------------------------
# ----------------------- Functions below from postproc.py ---------------------------
# ------------------------------------------------------------------------------------

'''compute_effective_sample_size and compute_group_auto_corr are from The Uncertainty Quantification Toolkit (UQTk)

Debusschere B, Sargsyan K, Safta C, Chowdhary K. The Uncertainty
Quantification Toolkit (UQTk). In: Ghanem R, Higdon D, Owhadi H, editors.
Handbook of Uncertainty Quantification. Springer; 2017. p. 1807–1827. Available
from: http://www.springer.com/us/book/9783319123844.
'''

def compute_effective_sample_size(n_sam,auto_corr):
    """Computes the effective sample size for a vector of samples
    by dividing the number of samples (n_sam) by the integral
    of the autocorrelation (auto_corr) between the samples. (i.e. the more
    correlated successive samples are, the less independent samples there are
    in the chain.)
    The algorithm is based on:
      Markov Chain Monte Carlo in Practice: A Roundtable Discussion
      Robert E. Kass, Bradley P. Carlin, Andrew Gelman and Radford M. Neal
      The American Statistician, Vol. 52, No. 2 (May, 1998), pp. 93-100
      Published by: American Statistical Association
      Article DOI: 10.2307/2685466
      Article Stable URL: http://www.jstor.org/stable/2685466
    """

    # Length of autocorrelation array
    n_ac = len(auto_corr)

    # Find the lag where the autocorrelation goes to zero (or below)
    i_zero = 1   # start at lag 1 since the autocorrelation has value 1.0 at lag 0 by definition
    done = False
    while (i_zero < n_ac and not done):
        if auto_corr[i_zero] > 0.0:
            i_zero += 1
        else:
            done = True

    went_to_zero = True
    if i_zero == n_ac:
        print("WARNING: Autocorrelation did not go to zero within range provided")
        went_to_zero = False
        

    # Integral relies on symmetry and the fact that autocorrelation is 1 at zero lag
    ESS = int(n_sam // (1.0+2.0*np.sum(auto_corr[1:i_zero])))

    return ESS, went_to_zero

def compute_group_auto_corr(v,maxlag):
    """Compute autocorrelation of v, an array where each column is a set of samples,
    for a lag ranging from 0 to maxlag-1. Ouputs numpy array with autocorrelation."""

    # Get dimensions of input array with samples
    n_pts = np.shape(v)[0]
    n_var = np.shape(v)[1]

    # Initialize array
    auto_corr = np.zeros((maxlag,n_var))

    # Get mean and variance of v for each variable over the samples provided
    v_m = v.mean(0)
    v_var = v.var(0)


    # Subtract the mean of v
    v_nm = v - v_m

    # Compute autocovariance of v over all variables
#     for lag in tqdm(range(maxlag)):
    for lag in range(maxlag):
        n_sum = n_pts - lag     # total number of terms in sum
        for i in range(n_sum):
            auto_corr[lag,:] += v_nm[i,:]*v_nm[i+lag,:]
        auto_corr[lag,:] /= float(n_sum)

    # Normalize by variance
    auto_corr /= v_var

    return auto_corr


# ------------------------------------------------------------------------------------
# ------------- Set up approximate zero-standard deviation pdf functions -------------
# ------------------------------------------------------------------------------------

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
        

# ------------------------------------------------------------------------------------
# -------------------- function to interpolate between two PDFs ----------------------
# ------------------------------------------------------------------------------------

def interp_KDE_1d(desired_param, unique_param_values, means, variances, stdevs, stdevs_approx,
                  KDE_fns, mesh_number, mesh_min, mesh_max, 
                  mesh_min_extended, mesh_max_extended, add_to_stdev, shape_param, type_, mean, renormalize=False, num_of_intervals = 5):
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
#         for subpop_ind in range(num_of_subpops):
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
#         for subpop_ind in range(num_of_subpops):
        if renormalize == False:
            pdf_fns[interval] = interpolate.interp1d(x_mesh, pdf[interval, :], axis = -1)
        elif renormalize == True:
            approx_integral = scipy.integrate.trapezoid(pdf[interval, :], x_mesh)
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