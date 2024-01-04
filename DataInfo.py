# Class DataInfo which stores metadata about the dataset
# Lincen's code includes a number of methods to get candidate cuts, but only this one is used

import numpy as np
import os
from datetime import datetime

class DataInfo:
    def __init__(self, X, Y, beam_width=None, alg_config=None, not_use_excl=None):
        pass
    
    def candidate_cuts_quantile_midpoints(self, num_candidate_cuts):
        """ Calculate the candidate cuts for each feature, using the quantile midpoints method.
        
        Parameters
        ---
        num_candidate_cuts : list or int
            Number of candidate cuts to use
            If this is a list, each feature's number of cuts is taken from the list
            If this is an int, that number is used for all features
        
        Returns
        ---
        candidate_cuts : Dict
            Dictionary of candidate cuts for each feature
        """