# Various functions for constraining the growth of a rule

import sys
import utils_calculating_cl

def validity_check(rule, icol, cut):
    """ Check whether a split is valid for a given rule

    
    Parameters
    ---
    rule : Rule object
        Rule to be checked
    icol : int
        Index of the column for which the candidate cut is calculated
    cut : float
        Candidate cut
    
    Returns
    ---
     : dict
        contains the validity including and excluding overlapping instances
    """
    pass

def check_split_validity(rule, icol, cut):
    """ Check validity when considering overlapping instances
    
    Parameters
    ---
    rule : Rule object
        Rule to be checked
    icol : int
        Index of the column for which the candidate cut is calculated
    cut : float
        Candidate cut
    
    Returns
    ---
    : bool
        Whether the split is valid
    """
    pass

def check_split_validity_excl(rule, icol, cut):
    """ Check validity without considering overlapping instances
    
    Parameters
    ---
    rule : Rule object
        Rule to be checked
    icol : int
        Index of the column for which the candidate cut is calculated
    cut : float
        Candidate cut
    
    Returns
    ---
    : bool
        Whether the split is valid
    """
    pass