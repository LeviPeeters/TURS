from numba import njit
import numpy as np

@njit
def calc_probs(target, num_class, smoothed=False):
    """ Given a vector containing the targets of a set of instances, calculate the probability of encountering each target.

    Parameters
    ---
    target : Array
        Vector containing the targets of a set of instances
    num_class : int
        Number of classes
    smoothed : bool
        Whether to smooth the target distibutions by adding one to each class count
    
    Returns
    ---
    probs : Array
        Probability of encountering each target
    """
    counts = np.bincount(target, minlength=num_class)
    if smoothed:
        counts = counts + np.ones(num_class, dtype=int) 
    return counts / np.sum(counts)

@njit
def calc_negloglike(p, n):
    """ Calculate the negative log-likelihood of a set of instances
    
    Parameters
    ---
    p : Array
        Vector containing the probability of encountering each target
    n : int
        Number of instances

    Returns
    ---
    negloglike : float
        Negative log-likelihood of the set of instances    
    """
    return -n * np.sum(np.log2(p[p !=0 ]) * p[p != 0])

