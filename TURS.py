# API

from datetime import datetime

import DataInfo
import Ruleset
import ModelEncoding
import DataEncoding
import utils_namedtuple
import utils

class TURS():
    def __init__(self,
                 num_candidate_cuts: int = 20,
                    max_num_rules: int = 500,
                    max_grow_iter: int = 500,
                    num_class_as_given: int = None,
                    beam_width: int = 10,
                    log_learning_process: bool = True,
                    log_folder_name: str = None,
                    dataset_name: str = None,
                    feature_names: list = None,
                    label_names: list = None,
                    which_features: list = None,
                    random_seed: int = None,
                    validity_check: str = "either"
                 ) -> None:
        
        self.alg_config = utils_namedtuple.AlgConfig(
            num_candidate_cuts=20, 
            max_num_rules=500, 
            max_grow_iter=500, 
            num_class_as_given=None,
            beam_width=10,
            log_learning_process=log_learning_process,
            log_folder_name=f"{datetime.now().strftime('%Y%m%d_%H%M')}_{dataset_name}",
            dataset_name=None,
            feature_names=feature_names,
            label_names=label_names,
            which_features=None,
            random_seed=None,
            validity_check="either"
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
            "ModelEncoding.*", 
            "DataEncoding.*", 
            "DataInfo.*", 
            "utils_calculating_cl.*",  
            "exp_predictive_perf.*",
            "run_uci.*",
        ]

    def fit(self, X_train, y_train, printing=True):
        data_info = DataInfo.DataInfo(X=X_train, y=y_train, beam_width=None, alg_config=self.alg_config)

        data_encoding = DataEncoding.NMLencoding(data_info)
        model_encoding = ModelEncoding.ModelEncodingDependingOnData(data_info)

        self.ruleset = Ruleset.Ruleset(
            data_info=data_info, 
            data_encoding=data_encoding, 
            model_encoding=model_encoding
        )

        self.ruleset.fit(max_iter=1000, printing=printing)
    
        return self.ruleset

    def generate_call_graph(self, X_train, y_train, filepath="call_graph.png"):
        data_info = DataInfo.DataInfo(X=X_train, y=y_train, beam_width=None, alg_config=self.alg_config)

        data_encoding = DataEncoding.NMLencoding(data_info)
        model_encoding = ModelEncoding.ModelEncodingDependingOnData(data_info)

        self.ruleset = Ruleset.Ruleset(
            data_info=data_info, 
            data_encoding=data_encoding, 
            model_encoding=model_encoding
        )

        utils.call_graph_filtered(self.ruleset.fit, filepath, custom_include=self.call_graph_custom_include)
        
        print("Call graph generated at", filepath)
    
    def predict(self, X_test):
        return self.ruleset.predict_ruleset(X_test)
    
    def predict_proba(self, X_test):
        return self.ruleset.predict_ruleset(X_test)

