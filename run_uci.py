import sys

import numpy as np
import time
from datetime import datetime

from sklearn.model_selection import StratifiedKFold

import utils_dataprep
import TURS

np.seterr(all='raise')
print("Running TURS with multithreading")

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

d = utils_dataprep.read_data(data_name)
X, y, class_labels, feature_names = utils_dataprep.preprocess_sparse(d)

kf = StratifiedKFold(n_splits=5, shuffle=True,
                     random_state=2)  # can also use sklearn.model_selection.StratifiedKFold
kfold = kf.split(X=np.zeros(shape=y.shape), y=y)# X is not actually required, and sparse matrices are not supported
kfold_list = list(kfold)

times = []
first_run = True # to avoid logging multiple folds


for fold in range(5):
    if fold_given is not None and fold != fold_given:
        continue
    print("running: ", data_name, "; fold: ", fold)

    X_train = X[kfold_list[fold][0], :]
    y_train = y[kfold_list[fold][0]]
    X_test = X[kfold_list[fold][1], :]
    y_test = y[kfold_list[fold][1]]

    start_time = time.time()
    
    turs = TURS.TURS(
        num_candidate_cuts=20,
        max_num_rules=500,
        max_grow_iter=500,
        num_class_as_given=None,
        beam_width=10,
        chunksize=97,
        log_learning_process=3 if first_run else 0,
        log_folder_name=datetime.now().strftime("%Y%m%d_%H%M%s") + "_" + data_name,
        dataset_name=None,
        feature_names=feature_names,
        which_features=None,
        random_seed=None,
        label_names=class_labels,
        validity_check="either"
    )

    turs.fit(X_train, y_train)

    end_time = time.time()
    times.append(end_time - start_time)

    first_run = False

print(f"Mean time: {np.mean(times)}")

