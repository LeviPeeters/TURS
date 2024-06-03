# API
from datetime import datetime

import DataInfo
import Ruleset
import utils_namedtuple
import os

class TURS():
    def __init__(self,
                 num_candidate_cuts: int = 20,
                    max_num_rules: int = 500,
                    max_grow_iter: int = 500,
                    num_class_as_given: int = None,
                    beam_width: int = 10,
                    chunksize: int = 8,
                    workers: int = -1,
                    log_learning_process: int = 1,
                    log_folder_name: str = None,
                    dataset_name: str = None,
                    feature_names: list = None,
                    label_names: list = None,
                    which_features: list = None,
                    random_seed: int = None,
                    validity_check: str = "either"
                 ) -> None:
        
        if log_folder_name is None:
            log_folder_name = f"{datetime.now().strftime('%Y%m%d_%H%M')}"
        
        if workers > os.cpu_count():
            workers = os.cpu_count()
            #print(f"Number of workers is set to {workers} as it exceeds the number of available CPUs.")
        
        self.alg_config = utils_namedtuple.AlgConfig(
            num_candidate_cuts=num_candidate_cuts, 
            max_num_rules=max_num_rules, 
            max_grow_iter=max_grow_iter, 
            num_class_as_given=num_class_as_given,
            beam_width=beam_width,
            chunksize=chunksize,
            workers=workers,
            log_learning_process=log_learning_process,
            log_folder_name=log_folder_name,
            dataset_name=dataset_name,
            feature_names=feature_names,
            label_names=label_names,
            which_features=which_features,
            random_seed=random_seed,
            validity_check=validity_check
        )

    def fit(self, X_train, y_train):
        data_info = DataInfo.DataInfo(X=X_train, y=y_train, beam_width=None, alg_config=self.alg_config)

        self.ruleset = Ruleset.Ruleset(
            data_info=data_info
        )

        self.ruleset.fit(max_iter=1000)
    
        return self.ruleset
    
    def predict(self, X_test):
        return self.ruleset.predict_ruleset(X_test)
    
    def predict_proba(self, X_test):
        return self.ruleset.predict_ruleset(X_test)

