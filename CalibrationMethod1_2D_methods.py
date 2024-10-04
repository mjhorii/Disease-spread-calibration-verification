import numpy as np
from scipy import interpolate
from math import sqrt
from scipy import stats
import scipy

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

def power(x, k):
    x_new = np.abs(x)**k
    x_new = x_new*np.sign(x)
    return x_new

def interp_KDE_1d(desired_param, unique_param_values, means, variances, stdevs, stdevs_approx,
                  KDE_fns, mesh_number, mesh_min, mesh_max, 
                  mesh_min_extended, mesh_max_extended, add_to_stdev, shape_param, type_, 
                  num_of_intervals, num_of_subpops, renormalize=False):
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
    
#     print(interpolated_var.shape)
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
#         reeval_interpolated_var[interval, subpop_ind] = interpolated_var[interval, subpop_ind] #will be 0
        reeval_interpolated_var[interval, subpop_ind] = (interpolated_stdev_approx[interval, subpop_ind])**2
        reeval_interpolated_stdev_approx[interval, subpop_ind] = interpolated_stdev_approx[interval, subpop_ind] #will be whatever the stdev approx value is
        if renormalize == False:
            pdf_fns[interval, subpop_ind], _ = return_zero_stdev_pdf(mean_val, shape_param, type_)
        elif renormalize == True:
            integral = return_integral_of_zero_stdev_pdf(mean_val, shape_param, type_, integral_range = (mesh_min_extended, mesh_max_extended))
            mean = 0 #for stdev = 0, mean will always be 0 in this use case
            pdf_fns[interval, subpop_ind], _ = return_zero_stdev_pdf(mean, shape_param, type_, renormalize = True, integral = integral)

    return x_mesh, pdf_fns, reeval_interpolated_mean, reeval_interpolated_var, np.squeeze(reeval_interpolated_stdev_approx)

def interp_KDE_2d(desired_jump_prob, desired_mobility, means, variances, stdevs, stdevs_approx, KDE_fns, num_of_intervals, num_of_subpops, unique_jumping_probs = np.linspace(0,0.001,10), unique_mobilities = np.linspace(0.005,0.025,17)):

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
                                                                                           501, -50, 150, -10000000, 10000000, 0.5, 0.4, "linear",
                                                                                           num_of_intervals, num_of_subpops,renormalize=True)

#     print('RUNNING INTERPOLATION BETWEEN MOBILITY AT JUMPING PROBABILITY ABOVE \n\n')
    x_mesh, pdf_jump_prob_above, mean_jump_prob_above, var_jump_prob_above, stdevs_approx_above = interp_KDE_1d(desired_mobility, unique_mobilities, 
                                                                                           np.swapaxes(means[:,:,jump_prob_above_ind,:],2,1),
                                                                                           np.swapaxes(variances[:,:,jump_prob_above_ind,:],2,1), 
                                                                                           np.swapaxes(stdevs[:,:,jump_prob_above_ind,:],2,1), 
                                                                                           np.swapaxes(stdevs_approx[:,:,jump_prob_above_ind,:],2,1),
                                                                                           KDE_fns[:,:,jump_prob_above_ind,:], 
                                                                                           501, -50, 150, -10000000, 10000000, 0.5, 0.4, "linear", 
                                                                                           num_of_intervals, num_of_subpops, renormalize=True)

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
                                KDEs_expanded_interpolated_by_mob, 201, 0, 100, 0, 100, 0.5, 0.5, "linear", num_of_intervals, num_of_subpops, renormalize = True)

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

def MCMC_KDE(sol, nburn, target = 0.95, unique_jumping_probs = np.linspace(0,0.001,10), unique_mobilities = np.linspace(0.005,0.025,17)):
    #target is confidence interval fraction to calculate bound for

    #---------------- Generate KDE -------------------
    data = np.array([sol['chain'][nburn:,0],
                    sol['chain'][nburn:,1]])

    kde = stats.kde.gaussian_kde(data)
    kde.set_bandwidth(bw_method='silverman')

    #--------------Calculate data to plot---------------
    plot_dim_x = 101 #height and width of plot in pixels
    plot_dim_y = 100 #height and width of plot in pixels

    xmin = unique_mobilities.min()
    xmax = unique_mobilities.max()
    ymin = unique_jumping_probs.min()
    ymax = unique_jumping_probs.max()

    # Regular grid to evaluate kde upon
    x_flat = np.linspace(xmin, xmax, plot_dim_x, endpoint=1)
    y_flat = np.linspace(ymin, ymax, plot_dim_y, endpoint=1)
    x,y = np.meshgrid(x_flat,y_flat)
    grid_coords = np.append(x.reshape(-1,1),y.reshape(-1,1),axis=1)
    
    print('grid_coords.shape', grid_coords.shape)

    Z,inside,inside_hollow,Z_renormalized=inv_cdf_2D(kde, grid_coords.T, x_flat, y_flat, target)

    marker=np.max(Z)*1.2
    Z_marked=np.where(inside_hollow==0,Z,marker)
    
    return Z, inside, inside_hollow, Z_renormalized, Z_marked, x_flat, y_flat

def inv_cdf_2D(kernel, positions, x_flat, y_flat, target):
    Z_vect_unnormalized=kernel(positions)
    # Normalize
    Z_vect=Z_vect_unnormalized/np.sum(Z_vect_unnormalized) 
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
    n_x = x_flat.shape[0]
    n_y = y_flat.shape[0]
    Z = np.reshape(Z_vect_unnormalized.T, (n_y, n_x))
    Z_unnormalized = np.reshape(Z_vect_unnormalized.T, (n_y, n_x))
    inside = np.reshape(inside_vect.T, (n_y, n_x))
    print('HERE',np.sum(Z))
    
    inside_pad=np.pad(inside,(1,1),'constant',constant_values=((1,1),(1,1))) # Add a border of 1s
    inside_hollow=np.copy(inside_pad)
    
    print('inside pad shape', inside_pad.shape)
    
    
    # Loop over interior points
    for ix in range(1,n_x-1+2):
        for iy in range(1,n_y-1+2):
            if np.sum(inside_pad[iy-1:iy+2, ix-1:ix+2])==9:
                inside_hollow[iy,ix]=0
    
    print('Z_unnormalized shape', Z_unnormalized.shape)
    print('yflat shape', y_flat.shape)
    print('xflat shape', x_flat.shape)
                
    approx_integral = scipy.integrate.trapezoid(Z_unnormalized, y_flat, axis = 0)
    approx_integral = scipy.integrate.trapezoid(approx_integral, x_flat, axis = 0)
    print('approx integral',approx_integral)
    Z_renormalized = Z_unnormalized/approx_integral
    print('max comparison', np.max(Z), np.max(Z_renormalized))
            
    return Z,inside,inside_hollow[1:-1,1:-1], Z_renormalized

def check_if_within_confidence_int(position, x_flat, y_flat, inside):
    x_ind = np.searchsorted(x_flat, position[0])
    y_ind = np.searchsorted(y_flat, position[1])
    
    print(x_ind, y_ind)
    
    if inside[y_ind, x_ind] == 1:
        if inside[y_ind, x_ind-1] == 1:
            if inside[y_ind-1, x_ind] == 1:
                if inside[y_ind-1, x_ind-1] ==1:
                    return True
    return False
    

'''
functions RemainderFixed and make_square_axes_with_colorbar from GitHub user ImportanceOfBeingErnest:
https://github.com/matplotlib/matplotlib/issues/15010#issuecomment-519721551
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

class RemainderFixed(axes_size.Scaled):
    def __init__(self, xsizes, ysizes, divider):
        self.xsizes =xsizes
        self.ysizes =ysizes
        self.div = divider

    def get_size(self, renderer):
        xrel, xabs = axes_size.AddList(self.xsizes).get_size(renderer)
        yrel, yabs = axes_size.AddList(self.ysizes).get_size(renderer)
        bb = Bbox.from_bounds(*self.div.get_position()).transformed(self.div._fig.transFigure)
        w = bb.width/self.div._fig.dpi - xabs
        h = bb.height/self.div._fig.dpi - yabs
        return 0, min([w,h])


def make_square_axes_with_colorbar(ax, size=0.1, pad=0.1):
    """ Make an axes square, add a colorbar axes next to it, 
        Parameters: size: Size of colorbar axes in inches
                    pad : Padding between axes and cbar in inches
        Returns: colorbar axes
    """
    divider = make_axes_locatable(ax)
    margin_size = axes_size.Fixed(size)
    pad_size = axes_size.Fixed(pad)
    xsizes = [pad_size, margin_size]
    yhax = divider.append_axes("right", size=margin_size, pad=pad_size)
    divider.set_horizontal([RemainderFixed(xsizes, [], divider)] + xsizes)
    divider.set_vertical([RemainderFixed(xsizes, [], divider)])
    return yhax

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