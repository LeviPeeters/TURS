import sys
import os

import numpy as np
import pandas as pd
import copy
import time
from datetime import datetime
import cProfile, pstats

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, auc, log_loss
from sklearn.model_selection import StratifiedKFold

def read_data(data_name):
    """ Read data from one of the standard datasets into a Pandas DataFrame

    Parameters
    ---
    data_name : str
        Relative filepath to dataset

    Returns
    ---
    DataFrame
        Containing the dataset
    """
    pass

def preprocess_data(d):
    """ TODO
    
    Parameters
    ---
    d : Dataframe
        Contains data that has just been read

    Returns
    ---
    d : DataFrame
        The same dataframe with some preprocessing steps performed

    """
    return d

def calculate_roc_auc_logloss(ruleset, y_test, y_pred_prob, y_train, y_pred_prob_train):
    """ Calculate various evaluation metrics using sklearn 
    
    Parameters
    ---
    ruleset : Ruleset object
        
    y_test : Array
        True labels for the test set
    y_pred_prob : Array
        Predicted probabilities for test set
    y_train : Array
        True labels for training set
    y_pred_prob_train : Array
        Predicted probabilities for train set

    Returns
    ---
    : list
        A list containing the evaluation metrics
    """
    pass

def calculate_rule_lengths(ruleset):
    """ Calculate the mean length of all rules in the rule set
    
    Parameters
    ---
    ruleset : Ruleset object
        
    Returns
    ---
     : float
        Mean length of rules

    """
    pass

def calculate_brier_and_prauc(ruleset, y_train, y_test, y_pred_prob, y_pred_prob_train):
    """ Calculate multi-class macro PR AUC, and (multi-class) Brier score
    
    Parameters
    ---
    ruleset : Ruleset object
        
    y_test : Array
        True labels for the test set
    y_pred_prob : Array
        Predicted probabilities for test set
    y_train : Array
        True labels for training set
    y_pred_prob_train : Array
        Predicted probabilities for train set

    Returns
    ---
    : list
        A list containing the evaluation metrics
    """
    pass

def calculate_train_test_prob_diff(ruleset, X_test, y_test):
    """ TODO
    
    Parameters
    ---
    ruleset : Ruleset object
        
    X_test: Array
        Features of test set
    y_test : Array
        True labels for the test set

    Returns
    ---
    train_test_prob_diff : float    
        TODO
    """
    pass

def calculate_random_picking_pred_performance(ruleset, X, y, num_repetition):
    """ Calculate the performance of a random picking strategy

    Parameters
    ---
    ruleset : Ruleset object
    X : Array
        Features of the dataset
    y : Array    
        True labels of the dataset
    num_repetition : int
        Number of repetitions
    
    Returns
    ---
    : list
        A list containing the evaluation metrics
    """    
    pass

def calculate_exp_res(ruleset, X_test, y_test, X_train, y_train, data_name, fold, start_time, end_time):
    """TODO
    
    Parameters
    ---
    ruleset : Ruleset object
    X_test : Array
        Features of test set
    y_test : Array
        True labels for the test set
    X_train : Array
        Features of train set
    y_train : Array
        True labels for the train set
    data_name : str
        Name of the dataset
    fold : int
        Fold number
    start_time : float
        Start time of the experiment
    end_time : float
        End time of the experiment

    Returns
    ---
    exp_res : dict
        A dictionary containing the evaluation metrics
    """
    pass

def run_(data_name, fold_given=None):
    """ Run the experiment for a given dataset and fold number
    
    Parameters
    ---
    data_name : str
        Name of the dataset
    fold_given : int
        Fold number

    Returns
    ---
    None
    """
    pass

if __name__ == "__main__":
    if len(sys.argv) == 3:
        run_(data_name=sys.argv[1], fold_given=int(sys.argv[2]))
    else:
        run_(data_name=sys.argv[1], fold_given=None)