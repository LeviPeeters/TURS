import numpy as np
import utils_predict

def cover_matrix(ruleset, X):
    """ Construct a cover matrix, indicating which rules cover which instances.
    
    Parameters
    ---
    ruleset : Ruleset object
        Ruleset object containing the rules
    X : Array
        Data matrix
    
    Returns
    ---
    cover_matrix : Array
        Cover matrix, indicating which rules cover which instances
    """
    pass

def predict_random_picking_for_overlaps(ruleset, X, seed):
    """ Predict the labels for the overlapping instances using a random picking strategy
    TODO: I'm not sure what this is used for
    
    Parameters
    ---
    ruleset : Ruleset object
        Ruleset object containing the rules
    X : Array
        Data matrix
    seed : int
        Seed for the random picking strategy
        
    Returns
    ---
    y_pred : Array
        Predicted probabilities
    """
    pass