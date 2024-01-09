import numpy as np
import utils_calculating_cl

def get_rule_local_prediction_for_unseen_data_this_rule_only(rule, X_test, y_test):
    """ This function computes the local prediction for a rule, given a test set.
    
    Parameters
    ----------
    rule : Rule
        The rule for which we want to compute the local prediction.
    X_test : np.array
        The test set.
    y_test : np.array
        The test labels.
    
    Returns
    -------
     : List
        Probability distributions and coverage for the prediction
    """

def get_rule_local_prediction_for_unseen_data(ruleset, X_test, y_test):
    """ This function computes the local prediction for a ruleset, given a test set.
    
    Parameters
    ----------
    ruleset : RuleSet
        The ruleset for which we want to compute the local prediction.
    X_test : np.array
        The test set.
    y_test : np.array
        The test labels.
    
    Returns
    -------
     : Dict
        Probability distributions and coverages for the prediction
    """

def predict_ruleset(ruleset, X_test, y_test):
    """ This function computes the local prediction using a ruleset, given a test set.
    
    Parameters
    ----------
    ruleset : RuleSet
        The ruleset for which we want to compute the local prediction.
    X_test : np.array
        The test set.
    y_test : np.array
        The test labels.
    
    Returns
    -------
     : Array
        Probability distributions for the prediction
    """
