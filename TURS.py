from datetime import datetime

import DataInfo
import Ruleset
import utils_namedtuple
import utils
import os

class TURS():
    def __init__(self,
                 num_candidate_cuts: int = 20,
                    max_num_rules: int = 500,
                    max_grow_iter: int = 500,
                    num_class_as_given: int = None,
                    beam_width: int = 10,
                    chunksize: int = 1,
                    workers: int = -1,
                    log_learning_process: int = 1,
                    log_folder_name: str = None,
                    model_folder_name: str = None,
                    dataset_name: str = None,
                    feature_names: list = None,
                    label_names: list = None,
                    which_features: list = None,
                    random_seed: int = None,
                    probability_threshold: bool = False,
                    force_else_50_50: bool = False,
                    validity_check: str = "either"
                 ) -> None:
        
        if log_folder_name is None:
            log_folder_name = f"{datetime.now().strftime('%Y%m%d_%H%M')}"
        if model_folder_name is None:
            print("Model folder name is not provided. Model will not be saved.")
        
        if workers > os.cpu_count():
            workers = os.cpu_count()
            print(f"Number of workers is set to {workers} as it exceeds the number of available CPUs.")
        elif workers == -1:
            print(f"Number of workers is set to -1, so TURS is not multiprocessed for this run.")
        
        if chunksize != 1:
            print("Chunksize is set to 1 for parallel expansion of rules")
            chunksize = 1
        
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
            model_folder_name=model_folder_name,
            dataset_name=dataset_name,
            feature_names=feature_names,
            label_names=label_names,
            which_features=which_features,
            random_seed=random_seed,
            probability_threshold=probability_threshold,
            force_else_50_50=force_else_50_50,
            validity_check=validity_check
        )

        self.call_graph_custom_include = [
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
            "DataInfo.*", 
            "utils_calculating_cl.*",  
            "exp_predictive_perf.*",
            "run_uci.*",
        ]

    def fit(self, X_train, y_train):

        data_info = DataInfo.DataInfo(X=X_train, y=y_train, beam_width=None, alg_config=self.alg_config)

        self.ruleset = Ruleset.Ruleset(
            data_info
        )

        folder_name = self.ruleset.fit(max_iter=1000)
        if folder_name is not None:
            self.alg_config = self.alg_config._replace(model_folder_name=folder_name)
    
        return self.ruleset

    def generate_call_graph(self, X_train, y_train, filepath="call_graph.png"):
        global data_info
        data_info = DataInfo.DataInfo(X=X_train, y=y_train, beam_width=None, alg_config=self.alg_config)

        self.ruleset = Ruleset.Ruleset(
            data_info=data_info, 
        )

        utils.call_graph_filtered(self.ruleset.fit, filepath, custom_include=self.call_graph_custom_include)
        
        print("Call graph generated at", filepath)

    
    def predict_probabilities(self, X_test):
        "Predict probabilities using the ruleset saved in the internal model folder"
        prediction_model = Ruleset.PredictUsingRuleset(self.alg_config.model_folder_name)
        return prediction_model.predict_ruleset(X_test)

