from scipy.special import gammaln
import numpy as np
from math import log

def log2comb(n, k):
    return (gammaln(n+1) - gammaln(n-k+1) - gammaln(k+1)) / log(2)

def universal_code_integers(value: int) -> float:
    """ computes the universal code of integers
    """
    const = 2.865064
    logsum = np.log2(const)
    if value == 0:
        logsum = 0
    elif value > 0:
        while True: # Recursive log
            value = np.log2(value)
            if value < 0.001:
                break
            logsum += value
    elif value < 0:
        raise ValueError('n should be larger than 0. The value was: {}'.format(value))
    return logsum