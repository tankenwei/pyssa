import numpy as np

from scipy.linalg import expm
from scipy.linalg import fractional_matrix_power as matpower
from scipy.stats import wishart

# Line search constants
LSALPHA = 0.5*(0.01+0.3)
LSBETA = 0.4
RDEC_THRESHOLD = 1e-8

def whitening(mat):
    """Returns a whitening matrix corresponding to the given covariance matrix"""
    W = matpower(mat,-0.5)
    assert np.isclose(np.imag(W),0).all()
    return np.real(W)

def random_rotation(d):
    """Returns a random rotation matrix of size (d x d)"""
    mat = np.random.normal(size=(d,d))
    mat = (mat - mat.T)/2
    return expm(mat)

def objective_function(params,mus,sigmas,epoch_sizes,M,k,calculate_grad = True,loss_translation = 0):
    """Returns loss, grad, rotation, mus_new, sigmas_new"""
    if M is None:
        Rcomplete = np.identity(params['dimensions_total'])
    else:
        Rcomplete = expm(M)
    assert np.isclose(Rcomplete.dot(Rcomplete.T), np.identity(params['dimensions_total'])).all()
    RScomplete = [Rcomplete.dot(sigma) for sigma in sigmas]
    RSRtcomplete = [RSc.dot(Rcomplete.T) for RSc in RScomplete]
    Rmucomplete = [Rcomplete.dot(mu) for mu in mus]
    RS = [RSc[:params['dimensions_stationary'],:] for RSc in RScomplete]
    RSRt = [RSRtc[:params['dimensions_stationary'],:params['dimensions_stationary']] for RSRtc in RSRtcomplete]
    #print(np.mean(RSRt, axis = 0))
    Rmu = [Rmuc[:params['dimensions_stationary']] for Rmuc in Rmucomplete]
    #print(np.mean(Rmu, axis = 0))
    losses = (-np.log(np.linalg.det(RSRt)) + np.power(Rmu,2).sum(axis =  1)) * epoch_sizes
    #print(losses)
    loss = np.sum(losses)
    #print(np.mean(np.linalg.det(RSRt),axis = 0))
    if loss_translation > 0:
        loss += loss_translation
    if loss < 0:
        if loss > -1e10:
            loss = 0
        else:
            raise ValueError(loss)
    normalized_loss = np.sqrt(2*loss) - np.sqrt(2*k-1)
    if not calculate_grad:
        return normalized_loss, None, Rcomplete, Rmucomplete, RSRtcomplete
    RSRt_inv = np.linalg.inv(RSRt)

    gradient = np.sum([(-a.dot(b) + c.reshape(params['dimensions_stationary'],1).dot(d.reshape(1,params['dimensions_total']))) * epoch_size 
                       for (a,b,c,d,epoch_size) in zip(RSRt_inv,RS,Rmu,mus,epoch_sizes)], axis = 0)
    gradient = 2 * np.vstack([gradient, np.zeros(shape = (params['dimensions_noisy'], params['dimensions_total']))])
    gradient = gradient.dot(Rcomplete.T) - Rcomplete.dot(gradient.T)

    normalized_gradient = gradient / np.sqrt(2*loss)
    return normalized_loss, normalized_gradient, Rcomplete, Rmucomplete, RSRtcomplete

def optimize_once(data, params, optNSources, loss_translation = 0, init = None):
    if init is None:
        sall = np.average(data['sigmas_initial'], axis = 0, weights = np.array(data['epoch_sizes']) - 1)
        init = random_rotation(params['dimensions_total']).dot(whitening(sall))
    B = init.copy()
    mus = [x.copy() - np.mean(data['mus_initial'], axis = 0) for x in data['mus_initial']]
    sigmas = [x.copy() for x in data['sigmas_initial']]

    mus = [B.dot(mu) for mu in mus]
    sigmas = [B.dot(sigma).dot(B.T) for sigma in sigmas]
    number_of_epochs = len(data['epoch_sizes'])
    if optNSources:
        k = number_of_epochs * params['dimensions_noisy'] * (params['dimensions_noisy'] + 3) / 2.
    else:
        k = number_of_epochs * params['dimensions_stationary'] * (params['dimensions_stationary'] + 3) / 2.
    maxiter = 10
    for i in range(maxiter):
        loss, grad, _, _, _ = objective_function(params,mus,sigmas,data['epoch_sizes'],None,k,loss_translation = loss_translation)
        if optNSources:
            loss *= -1
            grad *= -1
        if i == 0:
            alpha = -grad
        else:
            gamma = (grad*(grad - grad_old)).sum() / (grad_old ** 2).sum()
            alpha = -grad + alpha_old*gamma
        grad_old = grad.copy()
        alpha_old = alpha.copy()
        search = alpha / np.sqrt((alpha ** 2).sum() * 2)
        t = 1
        for i in range(10): # Line search
            M = t * search
            loss_new, _, rotation, mus_new, sigmas_new = objective_function(params,mus,sigmas,data['epoch_sizes'],M,k,calculate_grad = False,loss_translation = loss_translation)
            if optNSources:
                loss_new *= -1
            # Check if objective function has decreased enough
            if loss_new <= loss + LSALPHA*t*((0.5*grad*search).sum()): 
                break
            t *= LSBETA
        mus = mus_new
        sigmas = sigmas_new
        B = rotation.dot(B)
        if loss_new >= loss:
            break
        rel_decrease = abs((loss - loss_new)/loss)
        if rel_decrease < RDEC_THRESHOLD:
            break
    return loss,B

def optimize(data, dim_s, restarts = 10, optNSources = False, loss_translation = 0, init = None):
    """
    Performs SSA on the input data.
    
    Arguments
    ---------
        data            : A dictionary of mus_initial, sigmas_initial, epoch_sizes (generated by process_data)
        dim_s           : The number of stationary sources in the data
        restarts        : The number of restarts (to overcome local minima issues)
        optNSources     : Optimize the objective function with respect to the non-stationary sources (usually leave as False)
        loss_translation: Leave at 0 (by default); only supply a positive argument if there are issues with the objective function
        init            : Custom initialization matrix; defaults to whitening + random rotation.
                          For the algorithm to work, init must be a valid whitening matrix!
    """
    if init is not None and restarts > 1:
        print("A custom initialization matrix has been supplied; the optimizer need only be run once")
        restarts = 1
    params = {'dimensions_stationary':dim_s}
    params['dimensions_total'] = len(data['mus_initial'][0])
    params['dimensions_noisy'] = params['dimensions_total'] - params['dimensions_stationary']
    #params['number_of_epochs'] = len(params['epochs'])
    opt = (np.inf,None)
    try:
        for _ in range(restarts):
            buf = optimize_once(data, params, optNSources = optNSources, loss_translation = loss_translation, init = init)
            if buf[0] < opt[0]:
                opt = buf
    except ValueError as loss:
        t = loss.args[0]
        raise ValueError("The un-normalized loss function returned {}. This shouldn't happen, but a possible fix could be to run the optimizer again, and supply a positive argument to loss_translation".format(t))

    demixer = opt[1]
    print("Normalized loss function = {}".format(opt[0]))
    return demixer

def gen_toy_data(toy_data_params):
    """
    Generates a mixture of stationary and non-stationary multivariate normal data.
    toy_data_params is a dictionary containing:
        dimensions_noisy           : number of non-stationary sources
        mu_stationary              : vector of means of the stationary distribution)
        sigma_stationary (optional): covariance matrix of the stationary distribution
    If sigma_stationary is not supplied, one will be sampled from a Wishart(d,I) distribution
    where d is the number of stationary sources
    Returns the generated data and the mixing matrix.
    """
    dim_s = len(toy_data_params['mu_stationary'])
    dim_n = toy_data_params['dimensions_noisy']
    n_observations = sum(toy_data_params['epoch_sizes'])
    dim_total = dim_s + dim_n
    if 'sigma_stationary' not in toy_data_params:
        toy_data_params['sigma_stationary'] = wishart.rvs(dim_s,
                                                      np.identity(dim_s))
    
    if dim_s == 1:
        DATA_STATIONARY = np.random.normal(toy_data_params['mu_stationary'], 
                                           toy_data_params['sigma_stationary'], 
                                           size = n_observations).\
                                                  reshape(n_observations,1)
    else:
        DATA_STATIONARY = np.random.multivariate_normal(toy_data_params['mu_stationary'], 
                                                        toy_data_params['sigma_stationary'], 
                                                        size = n_observations)
    if dim_n == 1:
        DATA_NOISY = np.hstack([np.random.normal(np.random.normal(size = dim_n), 
                                                 np.random.uniform(size = dim_n), 
                                                 size = toy_data_params['epoch_size']) 
                                for epoch_size in toy_data_params['epoch_sizes']]).\
                                               reshape(n_observations,1)
    else:
        DATA_NOISY = np.vstack([np.random.multivariate_normal(np.random.normal(size = dim_n), 
                                                              wishart.rvs(dim_n,
                                                                          np.identity(dim_n)), 
                                                              size = epoch_size) 
                                for epoch_size in toy_data_params['epoch_sizes']])

    toy_data = np.hstack([DATA_STATIONARY,DATA_NOISY])
    mixer = random_rotation(dim_total)
    toy_data_mixed = toy_data.dot(mixer.T) # "Left" multiplication; actually vstacked mixer.dot(DATA[i])
    return toy_data_mixed, mixer

def process_data(data_raw, epochs):
    """
    Processes the data, to be used as input for the optimizer.
    
    Arguments
    ---------
        data_raw: An numpy array of observations (rows are observations, columns are sources)
        epochs  : Either an iterable of epoch sizes, or
                         an integer specifying the number of epochs (i.e. epochs are assumed to be equally sized)
                         
    returns a dictionary of:
        mus_initial   : A list of means of each epoch
        sigmas_initial: A list of covariance matrices of each epoch
        epoch_sizes   : A list of the epoch sizes
    """
    data = {}
    if hasattr(epochs, '__iter__'):
        data['epoch_sizes'] = epochs
    elif type(epochs) == int:
        data['epoch_sizes'] = [int(len(data_raw) / epochs)] * epochs
        remainder = len(data_raw) % epochs
        for i in range(remainder):
            data['epoch_sizes'][int(i*epochs/remainder)] += 1
    else:
        raise ValueError("epochs should either be an iterable or an integer!")
    assert sum(data['epoch_sizes']) == len(data_raw), "Sum of epoch sizes is not equal to number of observations!"
    epoch_slices = [slice(start,stop) 
                    for (start,stop) in zip(np.cumsum([0]+data['epoch_sizes'][:-1]),
                                            np.cumsum(data['epoch_sizes']))]
    data['mus_initial'] = [np.mean(data_raw[sl], axis = 0) for sl in epoch_slices]
    data['sigmas_initial'] = [np.cov(data_raw[sl], rowvar = False) for sl in epoch_slices]
    return data
