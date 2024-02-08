import sys
import os

import numpy as np
import pandas as pd
import copy
import time
from datetime import datetime
import cProfile, pstats

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, auc, log_loss
from sklearn.model_selection import StratifiedKFold

import DataInfo
import Ruleset
import utils_predict
import ModelEncoding
import DataEncoding

import exp_utils

def read_data(data_name):
    """ Read data from one of the standard datasets into a Pandas DataFrame
    TODO: All of the locatus related stuff should be moved to a separate file later

    Parameters
    ---
    data_name : str
        Relative filepath to dataset

    Returns
    ---
    DataFrame
        Containing the dataset
    """
    data_path = f"datasets/{data_name}.csv"
    
    # Some standard data sets have header rows, others don't
    datasets_without_header_row = ["chess", "iris", "waveform", "backnote", "contracept", "ionosphere",
                                   "magic", "car", "tic-tac-toe", "wine", "glass", "pendigits", "HeartCleveland"]
    datasets_with_header_row = ["avila", "anuran", "diabetes", "Vehicle", "DryBeans"]
    
    # Locatus data can be split into subsets for testing
    Locatus_data = ["Leiden", "Zeeland", "Amsterdam", "Zuid", "Full"]

    # # Some columns need to be converted to a different data type
    # dt_int = ["INW", "WVO", "PASSANTENAANTAL", "UNITID", "TARGET"]
    # dt_float = ["XCOORD", "YCOORD"]
    # dt_str = ["GEMEENTE", "PROVINCIE", "POSTCODE", "PC4", "WOONPLID", "WINKELGEBIEDID", "SUBCENTRUMID", "WINKELGEBIEDHOOFDTYPE", "WINKELGEBIEDTYPERING", "BINNENSTAD", "FORMULE", "GROEP", "BRANCHE", "HOOFDBRANCHE", "BEZOEKMOTIEFTYPE"]

    # For now, we only use a few columns
    dt_int = ["INW", "WVO", "PASSANTENAANTAL", "TARGET"]
    dt_float = []
    dt_str = ["GROEP", "BEZOEKMOTIEFTYPE"]

    if data_name in datasets_without_header_row:
        d = pd.read_csv(data_path, header=None)
        features = [f"X{i}" for i in range(d.shape[1] - 1)]
        features.append("y")
        d.columns = features
    elif data_name in datasets_with_header_row:
        d = pd.read_csv(data_path)
    elif data_name in Locatus_data:
        d = pd.read_csv("../Masterproject/data/prepared_data/merged_39278d2.csv", sep=",")
        
        # Select a subset of the data
        if data_name == 'Leiden':
            d = d[d["GEMEENTE"] == "Leiden"].copy()
        elif data_name == 'Zeeland':
            d = d[d["PROVINCIE"] == "Zeeland"].copy()
        elif data_name == 'Amsterdam':
            d = d[d["GEMEENTE"] == "Amsterdam"].copy()
        elif data_name == 'Zuid':
            d = d[d["PROVINCIE"] == "Zeeland"].copy() + d[d["PROVINCIE"] == "Limburg"].copy() + d[d["PROVINCIE"] == "Noord-Brabant"].copy()
            
        # Set correct dtypes and drop unneeded columns
        for col in d:
            if col in dt_int:
                d[col] = d[col].astype('int')
            elif col in dt_float:
                d[col] = d[col].astype('float')
            elif col in dt_str:
                d[col] = d[col].astype('str')
            else:
                d.drop(col, axis=1, inplace=True)

        
    else:
        sys.exit("error: data name not in the datasets lists that show whether the header should be included!")
    if data_name == "anuran":
        d = d.iloc[:, 1:]
    return d

def preprocess_data(d):
    """ Use a LabelEncoder to encode the labels and a OneHotEncoder to encode categorical features 
    Assumes that the labels are in the last column of the dataframe
    Columns with dtype float or int are left intact, columns with dtype str are one-hot encoded
    
    Parameters
    ---
    d : Dataframe
        Contains data that has just been read

    Returns
    ---
    d : DataFrame
        Preprocessed data

    """
    d = d.reset_index().drop("index", axis=1)

    le = LabelEncoder()
    d.iloc[:, -1] = le.fit_transform(d.iloc[:, -1])
    unique_labels = le.classes_

    ohe = OneHotEncoder(sparse_output=False, dtype=np.int, drop="if_binary")#, feature_name_combiner="concat")
    col_names = d.columns

    for icol in range(d.shape[1] - 1):
        if d.iloc[:, icol].dtype == "float" or d.iloc[:, icol].dtype == "int":
            # Numerical features are left intact
            d_transformed = d.iloc[:, icol]
            d_transformed.columns = [col_names[icol]]
        else:
            # One-hot encode the categorical features
            d_transformed = ohe.fit_transform(d.iloc[:, icol:(icol+1)])
            d_transformed = pd.DataFrame(d_transformed)
            d_transformed.columns = ohe.get_feature_names_out()

        # Add the transformed feature to the preprocessed dataframe
        if icol == 0:
            d_feature = d_transformed
        else:
            d_feature = pd.concat([d_feature, d_transformed], axis=1)

    # Add the labels to the preprocessed dataframe
    d = pd.concat([d_feature, d.iloc[:, -1]], axis=1)

    return d, unique_labels

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
