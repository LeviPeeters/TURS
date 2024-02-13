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

import TURS
import utils_dataprep

np.seterr(all='raise')
print("Running TURS with multithreading")

h = hpy()

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

d = utils_dataprep.read_data(data_name)
d, class_labels = utils_dataprep.preprocess_data(d)

X = d.iloc[:, :d.shape[1] - 1].to_numpy()
y = d.iloc[:, d.shape[1] - 1].to_numpy()

kf = StratifiedKFold(n_splits=5, shuffle=True,
                     random_state=2)  # can also use sklearn.model_selection.StratifiedKFold
kfold = kf.split(X=X, y=y)
kfold_list = list(kfold)

times = []
first_run = True

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
    
    turs = TURS(
        num_candidate_cuts=20,
        max_num_rules=500,
        max_grow_iter=500,
        num_class_as_given=None,
        beam_width=10,
        log_learning_process=first_run,
        log_folder_name=datetime.now().strftime("%Y%m%d_%H%M") + "_" + data_name,
        dataset_name=None,
        feature_names=None,
        label_names=None,
        which_features=None,
        random_seed=None,
        validity_check="either"
    )
    
    ruleset = turs.fit(X_train, y_train, printing=True)

    end_time = time.time()

    times.append(end_time - start_time)

    # Make predictions on test set
    y_pred = turs.predict(X_test)

    first_run = False

print("Average time:", np.mean(times))