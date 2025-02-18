import sys
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.sparse import hstack, csc_matrix

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
    data_path = f"datasets/{data_name}.csv"
    
    # Some standard data sets have header rows, others don't
    datasets_without_header_row = ["chess", "iris", "waveform", "backnote", "contracept", "ionosphere",
                                   "magic", "car", "tic-tac-toe", "wine", "glass", "pendigits", "HeartCleveland"]
    datasets_with_header_row = ["avila", "anuran", "diabetes", "Vehicle", "DryBeans"]

    if data_name in datasets_without_header_row:
        d = pd.read_csv(data_path, header=None)
        features = [f"X{i}" for i in range(d.shape[1] - 1)]
        features.append("y")
        d.columns = features
    elif data_name in datasets_with_header_row:
        d = pd.read_csv(data_path)        
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

def preprocess_sparse(df):
    """ Use a LabelEncoder to encode the labels and a OneHotEncoder to encode features 
    Assumes that the labels are in the last column of the dataframe
    Stores all features in a sparse matrix, including numerical ones
    TODO: It is better to store the numerical features in a dense matrix, as they are not sparse
    
    Parameters
    ---
    d : Dataframe
        Contains data that has just been read

    Returns
    ---
    d : NumPy array
        Preprocessed data

    """
    df = df.reset_index().drop("index", axis=1)

    le = LabelEncoder()
    y = le.fit_transform(df.iloc[:, -1])
    unique_labels = le.classes_
    feature_names = []

    ohe = OneHotEncoder(sparse_output=False, dtype=np.int, drop="if_binary")#, feature_name_combiner="concat")
    
    for icol in range(df.shape[1] - 1):
        if df.iloc[:, icol].dtype == "float" or df.iloc[:, icol].dtype == "int":
            # Numerical features are left intact
            d_transformed = csc_matrix(df.iloc[:, icol:(icol+1)])
            feature_names.append(df.columns[icol])
        else:
            d_transformed = ohe.fit_transform(df.iloc[:, icol:(icol+1)])
            feature_names.extend(ohe.get_feature_names_out())
            d_transformed = csc_matrix(d_transformed)

        # Add the transformed feature to the preprocessed dataframe
        if icol == 0:
            X = d_transformed
        else:
            X = hstack((X, d_transformed)) # Note, this is the scipy hstack, not the numpy hstack

    return X, y, list(unique_labels), feature_names