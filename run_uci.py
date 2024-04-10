import sys
import os
import pdb

import numpy as np
import pandas as pd
import copy
import time
from datetime import datetime
from guppy import hpy

from sklearn.model_selection import StratifiedKFold

import DataInfo 
import Ruleset
import ModelEncoding
import DataEncoding

import exp_predictive_perf
import utils_namedtuple
import utils

np.seterr(all='raise')

h = hpy()
make_call_graph = False
log_learning_process = True

exp_res_alldata = []
date_and_time = datetime.now().strftime("%Y%m%d_%H%M%S")

if len(sys.argv) == 3:
    data_name=sys.argv[1]
    fold_given=int(sys.argv[2])
elif len(sys.argv) == 2:
    data_name=sys.argv[1]
    fold_given=None
else:
    data_name = "iris"
    fold_given = 0

d = exp_predictive_perf.read_data(data_name)
d, class_labels = exp_predictive_perf.preprocess_data(d)

X = d.iloc[:, :d.shape[1] - 1].to_numpy()
y = d.iloc[:, d.shape[1] - 1].to_numpy()

kf = StratifiedKFold(n_splits=5, shuffle=True,
                     random_state=2)  # can also use sklearn.model_selection.StratifiedKFold
kfold = kf.split(X=X, y=y)
kfold_list = list(kfold)

times = []
first_run = True # to avoid logging multiple folds

for fold in range(5):
    if fold_given is not None and fold != fold_given:
        continue
    print("running: ", data_name, "; fold: ", fold)
    dtrain = copy.deepcopy(d.iloc[kfold_list[fold][0], :])
    dtest = copy.deepcopy(d.iloc[kfold_list[fold][1], :])

    X_train = dtrain.iloc[:, :dtrain.shape[1]-1].to_numpy()
    y_train = dtrain.iloc[:, dtrain.shape[1]-1].to_numpy()
    X_test = dtest.iloc[:, :-1].to_numpy()
    y_test = dtest.iloc[:, -1].to_numpy()

    start_time = time.time()
    alg_config = utils_namedtuple.AlgConfig(
        num_candidate_cuts=20, max_num_rules=500, max_grow_iter=500, num_class_as_given=None,
        beam_width=10,
        log_learning_process=True and first_run,
        log_folder_name=datetime.now().strftime("%Y%m%d_%H%M") + "_" + data_name,
        dataset_name=None,
        feature_names=d.columns[:-1],
        which_features=None,
        random_seed=None,
        label_names=class_labels,
        validity_check="either"
        )
    data_info = DataInfo.DataInfo(X=X_train, y=y_train, beam_width=None, alg_config=alg_config)

    data_encoding = DataEncoding.NMLencoding(data_info)
    model_encoding = ModelEncoding.ModelEncodingDependingOnData(data_info)
    ruleset = Ruleset.Ruleset(
        data_info=data_info, 
        data_encoding=data_encoding, 
        model_encoding=model_encoding
    )

    if make_call_graph:
        custom_include = [
            "Beam.*",
            "exp_utils.*",
            "ModellingGroup.*",
            "nml_regret.*",
            "Rule.*",
            "RuleGrowConstraint.*",
            "utils_modelencoding.*",
            "utils_namedtuple.*",
            "utils_predict.*",
            "Ruleset.*", 
            "ModelEncoding.*", 
            "DataEncoding.*", 
            "DataInfo.*", 
            "utils_calculating_cl.*",  
            "exp_predictive_perf.*",
            "run_uci.*",
        ]
        utils.call_graph_filtered(ruleset.fit, "call_graph.png", custom_include=custom_include)
    else:
        ruleset.fit(max_iter=1000)
    
    if first_run:
        print(ruleset)

    end_time = time.time()
    times.append(end_time - start_time)

    ## ROC_AUC and log-loss
    exp_res = exp_predictive_perf.calculate_exp_res(ruleset, X_test, y_test, X_train, y_train, data_name, fold, start_time, end_time)
    exp_res_alldata.append(exp_res)

    first_run = False

exp_res_df = pd.DataFrame(exp_res_alldata)
print(f"Mean time: {np.mean(times)}")

folder_name = "exp_uci_" + date_and_time[:8]
os.makedirs(folder_name, exist_ok=True)
if fold_given is None:
    res_file_name = "./" + folder_name + "/" + date_and_time + "_" + data_name + "_uci_datasets_res.csv"
else:
    res_file_name = "./" + folder_name + "/" + date_and_time + "_" + data_name + "_fold" + str(fold_given) + "_uci_datasets_res.csv"
exp_res_df.to_csv(res_file_name, index=False)
