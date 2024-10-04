'''Code from The Uncertainty Quantification Toolkit (UQTk).

Debusschere B, Sargsyan K, Safta C, Chowdhary K. The Uncertainty
Quantification Toolkit (UQTk). In: Ghanem R, Higdon D, Owhadi H, editors.
Handbook of Uncertainty Quantification. Springer; 2017. p. 1807–1827. Available
from: http://www.springer.com/us/book/9783319123844.
'''

import sys
import numpy as np
import scipy.stats
import scipy.linalg
import math
import uuid
import matplotlib.pyplot as plt
import arviz

#---------------------------------------------------------------------------------------
#  Adaptive Metropolis Markov Chain Monte Carlo
#---------------------------------------------------------------------------------------
def _ucov(spl,splmean,cov,lastup):
    r"""
    Update proposal covariance
    """
    if len(spl.shape) == 1:
        nspl = 1;
        ndim = spl.shape[0]
    else:
        (nspl,ndim)=spl.shape

    if nspl>0:
        for i in range(nspl):
            iglb    = lastup+i
            splmean = (iglb*splmean+spl[i])/(iglb+1)
            rt = (iglb-1.0)/iglb
            st = (iglb+1.0)/iglb**2
            cov = rt*cov+st*np.dot(np.reshape(spl[i]-splmean,(ndim,1)),np.reshape(spl[i]-splmean,(1,ndim)))

    return lastup + nspl, splmean, cov

def ammcmc(opts,likTpr,lpinfo,progress=True):
    r"""
    
    Adaptive Metropolis Markov Chain Monte Carlo

    
    Parameters
    ----------
    opts : dictionary of parameters

        - nsteps : no. of mcmc steps
        - nburn  : no. of mcmc steps for burn-in (proposal fixed to initial covariance)
        - nadapt : adapt every nadapt steps after nburn
        - nfinal : stop adapting after nfinal steps
        - inicov : initial covariance
        - coveps : small additive factor to ensure covariance matrix is positive definite (only added to diagonal if covariance matrix is singular without it)
        - burnsc : factor to scale up/down proposal if acceptance rate is too high/low
        - gamma  : factor to multiply proposed jump size with in the chain past the burn-in phase (Reduce this factor to get a higher acceptance rate. Defaults to 1.0)
        - spllo  : lower bounds for chain samples
        - splhi  : upper bounds for chain samples
        - rnseed : Optional seed for random number generator (needs to be integer >= 0) If not specified, then random number seed is not fixed and every chain will be different.
        - tmpchn : Optional; if present, will save chain state every 'ofreq' to ascii file. Filename is randomly generated if tmpchn is set to 'tmpchn', or set to the string passed through this option if not present, chain states are not saved during the MCMC progress

    cini : starting mcmc state

    likTpr : log-posterior function; it takes two input parameters as follows
        - first parameter is a 1D array containing the chain state at which the posterior will to be evaluated
        - the second parameter contains settings the user can pass to this function; see below info for 'lpinfo'
        - this function is expected to return log-Likelihood and log-Prior values (in this order)

    lpinfo : info to be passed to the log-posterior function

        this object can be of any type (e.g. None, scalar, list, array, dictionary, etc) 
        as long as it is consistent with settings expected inside the 'likTpr' function

    Returns
    -------
    mcmcRes: results dictionary

        - 'chain' : chain samples (nsteps x chain dimension)
        - 'cmap' :  MAP estimate 
        - 'pmap' :  MAP log posterior
        - 'accr' :  overall acceptance rate
        - 'accb' : fraction of samples inside bounds
        - 'rejAll' : overall no. of samples rejected
        - rejOut' : no. of samples rejected due to being outside bounds
        - 'minfo' : meta_info, acceptance probability, log likelihood, log prior
        - 'final_cov' : the covariance matrix at the end of the run
    """
    # -------------------------------------------------------------------------------
    # Parse options
    # -------------------------------------------------------------------------------
    nsteps = opts['nsteps']
    nburn  = opts['nburn' ]
    nadapt = opts['nadapt']
    nfinal = opts['nfinal']
    inicov = opts['inicov']
    coveps = opts['coveps']
    burnsc = opts['burnsc']
    spllo  = opts['spllo' ]
    splhi  = opts['splhi' ]
    cini   = opts['inistate']

    if 'gamma' not in opts:
        gamma = 1.0 # default
    else:
        gamma  = opts['gamma' ]

    if 'ofreq' not in opts:
        ofreq = 10000 # default
    else:
        ofreq  = opts['ofreq' ]

    if 'logfile' not in opts:
        log_file = 'logMCMC.txt'
    else:
        log_file = opts['logfile']

    f = open(log_file, "w")
    f.write("nsteps: %d\n"%(nsteps))
    f.write("gamma: %f\n"%(gamma))
    f.close()

    if 'tmpchn' not in opts:
        tmp_file = 'None'
    else:
        if opts['tmpchn'] == 'tmpchn':
            tmp_file = str(uuid.uuid4())+'.dat'
        else:
            tmp_file = opts['tmpchn']
        f = open(log_file, "a")
        f.write('Saving intermediate chains to %s\n'%(tmp_file))
        f.close()

    # If desired, fix random number seed to make chain reproducible
    if 'rnseed' in opts:
        iseed = opts['rnseed']
        if isinstance(iseed, int) and iseed >= 0:
            np.random.seed(iseed)
            print('\nam-mcmc::Fixing the random number seed to ', iseed)
        else:
            print('\nWARNING: am-mcmc::invalid random number seed specified: ', iseed)
            print('Will proceed without fixing random number seed.\n')

    rej    = 0;                        # Counts number of samples rejected
    rejlim = 0;                        # Counts number of samples rejected as out of prior bounds
    rejsc  = 0;                        # Counts number of rejected samples since last rescaling
    # -------------------------------------------------------------------------------
    # Pre-processing
    # -------------------------------------------------------------------------------
    cdim   = cini.shape[0]             # chain dimensionality
    cov    = np.zeros((cdim,cdim))    # covariance matrix
    spls   = np.zeros((nsteps,cdim))  # MCMC samples
    meta_info = np.zeros((nsteps,3))  # Column for acceptance probability and posterior prob. of current sample
    na     = 0                         # counter for accepted jumps
    sigcv  = 2.4*gamma/np.sqrt(cdim)  # covariance factor
    spls[0] = cini                     # initial sample set
    p1LikPri = likTpr(spls[0],lpinfo)  # and posterior probability of initial sample set
    if not isinstance(p1LikPri, list):
        print('\nERROR: This version requires the model return both log-likelihood and log-prior:',p1LikPri)
        return {}
    p1 = p1LikPri[0]+p1LikPri[1]
    meta_info[0] = [0.e0,p1LikPri[0],p1LikPri[1]]     # Arbitrary initial acceptance and posterior probability of initial guess
    pmode = p1                         # store current chain MAP probability value
    cmode = spls[0]                    # current MAP parameter Set
    nref  = 0                          # Samples since last proposal rescaling
    # -------------------------------------------------------------------------------
    # optional progress bar
    # -------------------------------------------------------------------------------
    if progress == True:
        try:
            from tqdm import tqdm
            mcmc_iterator = tqdm(range(nsteps-1))
        except:
            print("*For a progress bar, please install tqdm.")
            mcmc_iterator = range(nsteps-1)
    # -------------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------------
    for k in mcmc_iterator:
        # Deal with covariance matrix
        if k == 0:
            splmean   = spls[0];
            propcov   = inicov ;
            Rchol     = scipy.linalg.cholesky(propcov) ;
            lastup    = 1;      # last covariance update
        else:
            if (nadapt>0) and ((k+1)%nadapt)==0:
                if k<nburn:
                    if float(rejsc)/nref>0.95:
                        Rchol = Rchol/burnsc # scale down proposal
                        print('   Scaling down the proposal at step %d'%(k))
                    elif float(rejsc)/nref<0.05:
                        Rchol = Rchol*burnsc # scale up proposal
                        print('   Scaling up the proposal at step %d'%(k))
                    nref  = 0 ;
                    rejsc = 0 ;
                else:
                    lastup,splmean,cov=_ucov(spls[lastup:lastup+nadapt,:],splmean,cov,lastup)
                    try:
                        Rchol = scipy.linalg.cholesky(cov)
                    except scipy.linalg.LinAlgError:
                        try:
                            # add to diagonal to make the matrix positive definite
                            Rchol = scipy.linalg.cholesky(cov+coveps*np.identity(cdim))
                        except scipy.linalg.LinAlgError:
                            print('WARNING: Covariance matrix is singular even after the correction')
                    Rchol = Rchol*sigcv
        #-Done with covariance matrix
        nref = nref + 1
        #
        # generate proposal and check bounds
        #
        u  = spls[k]+np.dot(np.random.randn(1,cdim),Rchol)[0];
        if np.any(np.less(u,spllo)) or np.any(np.greater(u,splhi)):
            outofbound = True
            accept     = False
            p2 = -1.e100        # Arbitrarily low posterior likelihood
            pr = -1.e100        # Arbitrarily low acceptance probability
        else:
            outofbound = False
        if not outofbound:
            p2Lik,p2Pri = likTpr(u,lpinfo)
            p2 = p2Lik+p2Pri
            pr = np.exp(p2-p1);
            if (pr>=1.0) or (np.random.random_sample()<=pr):
                spls[k+1] = u.copy();                # Store accepted sample
                meta_info[k+1] = [pr,p2Lik,p2Pri]    # and its meta information
                p1 = p2;
                if p1 > pmode:
                    pmode = p1 ;
                    cmode = spls[k+1] ;
                accept = True
            else:
                accept = False
        #
        # See if we can do anything about a rejected proposal
        #
        if not accept:
            # if 'am' then reject
            spls[k+1]=spls[k];
            meta_info[k+1,0] = pr               # acceptance probability of failed sample
            meta_info[k+1,1:] = meta_info[k,1:] # Posterior probability of sample k that has been retained
            rej   = rej   + 1;
            rejsc = rejsc + 1;
            if outofbound:
                rejlim  = rejlim + 1;
            # Done with if over methods
        # Done with if over original accept
        if ((k+2)%ofreq==0 and tmp_file != 'None'):
            f = open(log_file, "a")
            acc_rate = float(k+1)/float(k+1+rej)*100
            f.write("No. steps: %d, No. of rej:%d, acc. rate:%d%%\n"%(k+1,rej,round(acc_rate)))
            f.write("Geweke scores: ")
            #Calculate geweke scores:
            for i in range(cdim): #for each chain dimension,
                f.write("%f, "%(arviz.geweke(spls[:k+1,i], first=0.1, last=0.5, intervals=1)[0][1]))
            f.write("\n")
            f.close()
            fout = open(tmp_file, 'ab')
            dataout = np.concatenate((spls[k-ofreq+1:k+1,:], meta_info[k-ofreq+1:k+1,1:]), axis=1)
            np.savetxt(fout, dataout, fmt='%.8e',delimiter=' ', newline='\n')
            fout.close()
    # Done loop over all steps

    # return output dictionary: samples, MAP sample and its posterior probability, overall acceptance probability
    # and probability of having sample inside prior bounds, overall number of samples rejected, and rejected
    # due to being out of bounds.
    mcmcRes={}
    mcmcRes['chain' ] = spls                     # chain
    mcmcRes['cmap'  ] = cmode                    # MAP state
    mcmcRes['pmap'  ] = pmode                    # MAP log posterior
    mcmcRes['accr'  ] = 1.0-float(rej)/nsteps    # acceptance rate (overall)
    mcmcRes['accb'  ] = 1.0-float(rejlim)/nsteps # samples inside bounds
    mcmcRes['rejAll'] = rej                      # overall no. of samples rejected
    mcmcRes['rejOut'] = rejlim                   # no. of samples rejected due to being outside bounds
    mcmcRes['minfo' ] = meta_info                # acceptance probability, log likelihood, log prior
    mcmcRes['final_cov'] = cov                   # the covariance matrix at the end of the run.
    return mcmcRes

def ammcmc_resume_run(opts,likTpr,lpinfo,progress=True):
    r"""
    
    Adaptive Metropolis Markov Chain Monte Carlo

    ALTERED TO BE ABLE TO RESUME A PREVIOUS MCMC RUN
    - requires previous mcmcRes

    
    Parameters
    ----------
    opts : dictionary of parameters

        - nsteps : no. of mcmc steps
        - nburn  : no. of mcmc steps for burn-in (proposal fixed to initial covariance)
        - nadapt : adapt every nadapt steps after nburn
        - nfinal : stop adapting after nfinal steps
        - inicov : initial covariance
        - coveps : small additive factor to ensure covariance matrix is positive definite (only added to diagonal if covariance matrix is singular without it)
        - burnsc : factor to scale up/down proposal if acceptance rate is too high/low
        - gamma  : factor to multiply proposed jump size with in the chain past the burn-in phase (Reduce this factor to get a higher acceptance rate. Defaults to 1.0)
        - spllo  : lower bounds for chain samples
        - splhi  : upper bounds for chain samples
        - rnseed : Optional seed for random number generator (needs to be integer >= 0) If not specified, then random number seed is not fixed and every chain will be different.
        - tmpchn : Optional; if present, will save chain state every 'ofreq' to ascii file. Filename is randomly generated if tmpchn is set to 'tmpchn', or set to the string passed through this option if not present, chain states are not saved during the MCMC progress
        - prevmcmcRes : previous results from amcmc function

    cini : starting mcmc state

    likTpr : log-posterior function; it takes two input parameters as follows
        - first parameter is a 1D array containing the chain state at which the posterior will to be evaluated
        - the second parameter contains settings the user can pass to this function; see below info for 'lpinfo'
        - this function is expected to return log-Likelihood and log-Prior values (in this order)

    lpinfo : info to be passed to the log-posterior function

        this object can be of any type (e.g. None, scalar, list, array, dictionary, etc) 
        as long as it is consistent with settings expected inside the 'likTpr' function

    Returns
    -------
    mcmcRes: results dictionary

        - 'chain' : chain samples (nsteps x chain dimension)
        - 'cmap' :  MAP estimate 
        - 'pmap' :  MAP log posterior
        - 'accr' :  overall acceptance rate
        - 'accb' : fraction of samples inside bounds
        - 'rejAll' : overall no. of samples rejected
        - rejOut' : no. of samples rejected due to being outside bounds
        - 'minfo' : meta_info, acceptance probability, log likelihood, log prior
        - 'final_cov' : the covariance matrix at the end of the run
    """
    # -------------------------------------------------------------------------------
    # Parse options
    # -------------------------------------------------------------------------------
    prev_mcmcRes = opts['prevmcmcRes']
    nsteps = opts['nsteps']
    # nburn  = opts['nburn' ]
    nadapt = opts['nadapt']
    nfinal = opts['nfinal']
    # inicov = opts['inicov']
    coveps = opts['coveps']
    burnsc = opts['burnsc']
    spllo  = opts['spllo' ]
    splhi  = opts['splhi' ]
    # cini   = opts['inistate']
    inicov = prev_mcmcRes['final_cov'] #Set new initial covariance to be previous final covariance
    cini = prev_mcmcRes['chain'][-1] #Set new initial chain location to be previous final chain location
    nburn = 0 #Start adaptation right away

    if 'gamma' not in opts:
        gamma = 1.0 # default
    else:
        gamma  = opts['gamma' ]

    if 'ofreq' not in opts:
        ofreq = 10000 # default
    else:
        ofreq  = opts['ofreq' ]

    if 'logfile' not in opts:
        log_file = 'logMCMC.txt'
    else:
        log_file = opts['logfile']

    f = open(log_file, "w")
    f.write("nsteps: %d\n"%(nsteps))
    f.write("gamma: %f\n"%(gamma))
    f.close()

    if 'tmpchn' not in opts:
        tmp_file = 'None'
    else:
        if opts['tmpchn'] == 'tmpchn':
            tmp_file = str(uuid.uuid4())+'.dat'
        else:
            tmp_file = opts['tmpchn']
        f = open(log_file, "a")
        f.write('Saving intermediate chains to %s\n'%(tmp_file))
        f.close()

    # If desired, fix random number seed to make chain reproducible
    if 'rnseed' in opts:
        iseed = opts['rnseed']
        if isinstance(iseed, int) and iseed >= 0:
            np.random.seed(iseed)
            print('\nam-mcmc::Fixing the random number seed to ', iseed)
        else:
            print('\nWARNING: am-mcmc::invalid random number seed specified: ', iseed)
            print('Will proceed without fixing random number seed.\n')

    rej    = 0;                        # Counts number of samples rejected
    rejlim = 0;                        # Counts number of samples rejected as out of prior bounds
    rejsc  = 0;                        # Counts number of rejected samples since last rescaling
    # -------------------------------------------------------------------------------
    # Pre-processing
    # -------------------------------------------------------------------------------
    cdim   = cini.shape[0]             # chain dimensionality
    # cov    = np.zeros((cdim,cdim))    # covariance matrix
    cov = inicov                        #starting adaptive covariance matrix is EQUAL to previous final covariance matrix
    spls   = np.zeros((nsteps,cdim))  # MCMC samples
    meta_info = np.zeros((nsteps,3))  # Column for acceptance probability and posterior prob. of current sample
    na     = 0                         # counter for accepted jumps
    sigcv  = 2.4*gamma/np.sqrt(cdim)  # covariance factor
    spls[0] = cini                     # initial sample set
    p1LikPri = likTpr(spls[0],lpinfo)  # and posterior probability of initial sample set
    if not isinstance(p1LikPri, list):
        print('\nERROR: This version requires the model return both log-likelihood and log-prior:',p1LikPri)
        return {}
    p1 = p1LikPri[0]+p1LikPri[1]
    meta_info[0] = [0.e0,p1LikPri[0],p1LikPri[1]]     # Arbitrary initial acceptance and posterior probability of initial guess
    pmode = p1                         # store current chain MAP probability value
    cmode = spls[0]                    # current MAP parameter Set
    nref  = 0                          # Samples since last proposal rescaling
    # -------------------------------------------------------------------------------
    # optional progress bar
    # -------------------------------------------------------------------------------
    if progress == True:
        try:
            from tqdm import tqdm
            mcmc_iterator = tqdm(range(nsteps-1))
        except:
            print("*For a progress bar, please install tqdm.")
            mcmc_iterator = range(nsteps-1)
    # -------------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------------
    for k in mcmc_iterator:
        # Deal with covariance matrix
        if k == 0:
            splmean   = spls[0];
            propcov   = inicov ;
            Rchol     = scipy.linalg.cholesky(propcov) ;
            # lastup = 1                    # last covariance update
            lastup = mcmcRes['chain'].shape[0];      # the last covariance update was from the previous run
        else:
            if (nadapt>0) and ((k+1)%nadapt)==0:
                if k<nburn:
                    if float(rejsc)/nref>0.95:
                        Rchol = Rchol/burnsc # scale down proposal
                        print('   Scaling down the proposal at step %d'%(k))
                    elif float(rejsc)/nref<0.05:
                        Rchol = Rchol*burnsc # scale up proposal
                        print('   Scaling up the proposal at step %d'%(k))
                    nref  = 0 ;
                    rejsc = 0 ;
                else:
                    lastup,splmean,cov=_ucov(spls[lastup:lastup+nadapt,:],splmean,cov,lastup)
                    try:
                        Rchol = scipy.linalg.cholesky(cov)
                    except scipy.linalg.LinAlgError:
                        try:
                            # add to diagonal to make the matrix positive definite
                            Rchol = scipy.linalg.cholesky(cov+coveps*np.identity(cdim))
                        except scipy.linalg.LinAlgError:
                            print('WARNING: Covariance matrix is singular even after the correction')
                    Rchol = Rchol*sigcv
        #-Done with covariance matrix
        nref = nref + 1
        #
        # generate proposal and check bounds
        #
        u  = spls[k]+np.dot(np.random.randn(1,cdim),Rchol)[0];
        if np.any(np.less(u,spllo)) or np.any(np.greater(u,splhi)):
            outofbound = True
            accept     = False
            p2 = -1.e100        # Arbitrarily low posterior likelihood
            pr = -1.e100        # Arbitrarily low acceptance probability
        else:
            outofbound = False
        if not outofbound:
            p2Lik,p2Pri = likTpr(u,lpinfo)
            p2 = p2Lik+p2Pri
            pr = np.exp(p2-p1);
            if (pr>=1.0) or (np.random.random_sample()<=pr):
                spls[k+1] = u.copy();                # Store accepted sample
                meta_info[k+1] = [pr,p2Lik,p2Pri]    # and its meta information
                p1 = p2;
                if p1 > pmode:
                    pmode = p1 ;
                    cmode = spls[k+1] ;
                accept = True
            else:
                accept = False
        #
        # See if we can do anything about a rejected proposal
        #
        if not accept:
            # if 'am' then reject
            spls[k+1]=spls[k];
            meta_info[k+1,0] = pr               # acceptance probability of failed sample
            meta_info[k+1,1:] = meta_info[k,1:] # Posterior probability of sample k that has been retained
            rej   = rej   + 1;
            rejsc = rejsc + 1;
            if outofbound:
                rejlim  = rejlim + 1;
            # Done with if over methods
        # Done with if over original accept
        if ((k+2)%ofreq==0 and tmp_file != 'None'):
            f = open(log_file, "a")
            acc_rate = float(k+1)/float(k+1+rej)*100
            f.write("No. steps: %d, No. of rej:%d, acc. rate:%d%%\n"%(k+1,rej,round(acc_rate)))
            f.write("Geweke scores: ")
            #Calculate geweke scores:
            for i in range(cdim): #for each chain dimension,
                f.write("%f, "%(arviz.geweke(spls[:k+1,i], first=0.1, last=0.5, intervals=1)[0][1]))
            f.write("\n")
            f.close()
            fout = open(tmp_file, 'ab')
            dataout = np.concatenate((spls[k-ofreq+1:k+1,:], meta_info[k-ofreq+1:k+1,1:]), axis=1)
            np.savetxt(fout, dataout, fmt='%.8e',delimiter=' ', newline='\n')
            fout.close()
    # Done loop over all steps

    # return output dictionary: samples, MAP sample and its posterior probability, overall acceptance probability
    # and probability of having sample inside prior bounds, overall number of samples rejected, and rejected
    # due to being out of bounds.
    mcmcRes={}
    # mcmcRes['chain' ] = spls                     # chain
    mcmcRes['chain' ] = np.concatenate((prev_mcmcRes['chain'], spls))  # chain, including the values from the previous run
    mcmcRes['cmap'  ] = cmode                    # MAP state
    mcmcRes['pmap'  ] = pmode                    # MAP log posterior
    mcmcRes['accr'  ] = 1.0-float(rej)/nsteps    # acceptance rate (overall)
    mcmcRes['accb'  ] = 1.0-float(rejlim)/nsteps # samples inside bounds
    mcmcRes['rejAll'] = rej                      # overall no. of samples rejected
    mcmcRes['rejOut'] = rejlim                   # no. of samples rejected due to being outside bounds
    mcmcRes['minfo' ] = meta_info                # acceptance probability, log likelihood, log prior
    mcmcRes['final_cov'] = cov                   # the covariance matrix at the end of the run.
    return mcmcRes

def save_mcmc_chain(fout,mcmc_chain,filetype="h5"):
    """
    save mcmc output
    """
    if filetype=="h5":
        import h5py
        f = h5py.File(fout, "w")
        dset = f.create_dataset("chain",     data=mcmc_chain['chain'],     compression="gzip")
        mdIn = f.create_dataset("mode",      data=mcmc_chain['cmap'],      compression="gzip")
        mdPs = f.create_dataset("modepost",  data=[mcmc_chain['pmap']],    compression="gzip")
        minf = f.create_dataset("minfo",     data=mcmc_chain['minfo'],     compression="gzip")
        accr = f.create_dataset("accr",      data=[mcmc_chain['accr']],    compression="gzip")
        fcov = f.create_dataset("final_cov", data=mcmc_chain['final_cov'], compression="gzip")
        f.close()
    elif filetype=="pkl":
        output = open(fout, 'wb')
        pickle.dump(mcmc_chain, output)
        output.close()
    else:
        print("save_mcmc_chain(): Unknown filetype ",filetype," ->quit!")
        sys.exit()

def test_banana(nsamples, seed=100, a=1., b=100.):
    """ Sampling a 2d highly correlated Gaussian distribution """

    def rosenbrock_2d(theta, tmparg):
        """
        Rosenbrock function in 2D
        """

        x = theta[0]
        y = theta[1]

        logprob = -(x - a)**2 - b * (y - x**2)**2
        return [logprob, 0.0]

    nburn = 1000
    nskip = 10000
    nthin = 100
    opts = {"nsteps": nsamples, "nfinal": 10000000,"gamma": 0.2,
            "inicov": np.array([[0.0001,0],[0,0.0001]]),"inistate": np.array([-0.74976547, 1.3426804]),
            "spllo": np.array([-100,-100]),"splhi": np.array([[100,100]]),
            "logfile": "tmp1.log","burnsc":5,
            "nburn":nburn,"nadapt":100,"coveps":1.e-10,"ofreq":5000,"tmpchn":"tmpchn"
            }

    ndim = 2
    np.random.seed(seed)
    theta0 = np.random.normal(1, 1, ndim)
    print(f'Start point:{theta0}')
    opts["inistate"] = theta0

    print('Sampling 2D Rosenbrock function with AMCMC ...')
    sol=ammcmc(opts,rosenbrock_2d,None)
    samples = sol['chain']
    logprob = sol['minfo'][:,1]

    import matplotlib.pyplot as plt
    samples = samples[nskip::nthin]
    logprob = logprob[nskip::nthin]

    print('Acceptance rate',sol['accr'])
    print('Mean:',np.mean(samples, axis=0))
    print('Var:',np.var(samples, axis=0))
    print('Cov:',np.cov(samples.T))

    plt.subplot(1,3,1)
    plt.scatter(samples[:,0], samples[:,1], s=4, c = np.exp(logprob))

    plt.subplot(1,3,2)
    plt.plot(samples[:, 0], logprob, ',')
    plt.xlabel("x-samples")

    plt.subplot(1,3,3)
    plt.plot(samples[:, 1], logprob, ',')
    plt.xlabel("y-samples")
    plt.savefig("amcmc_rosenbrock.pdf")
    # plt.show()
    plt.close()

if __name__ == "__main__":

    nspls = int(sys.argv[1])
    a, b = 1, 100
    if len(sys.argv)>2:
        b = float(sys.argv[2])

    test_banana(nspls, a=a, b=b)

