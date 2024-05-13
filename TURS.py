# API

from datetime import datetime
import multiprocessing as mp
from multiprocessing.managers import BaseManager, NamespaceProxy
import inspect

import DataInfo
import Ruleset
import ModelEncoding
import DataEncoding
import utils_namedtuple
import utils

"""
class MyManager(BaseManager): pass

class ProxyBase(NamespaceProxy):
    _exposed_ = ('__getattribute__', '__setattr__', '__delattr__')

class DataInfoProxy(ProxyBase): pass


def register_proxy(name, cls, proxy):
    for attr in dir(cls):
        if inspect.ismethod(getattr(cls, attr)) and not attr.startswith("__"):
            proxy._exposed_ += (attr,)
            setattr(proxy, attr, 
                    lambda s: object.__getattribute__(s, '_callmethod')(attr))
    MyManager.register(name, cls, proxy)

register_proxy('DataInfo', DataInfo.DataInfo, DataInfoProxy)"""


class TURS():
    def __init__(self,
                 num_candidate_cuts: int = 20,
                    max_num_rules: int = 500,
                    max_grow_iter: int = 500,
                    num_class_as_given: int = None,
                    beam_width: int = 10,
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
        
        self.alg_config = utils_namedtuple.AlgConfig(
            num_candidate_cuts=num_candidate_cuts, 
            max_num_rules=max_num_rules, 
            max_grow_iter=max_grow_iter, 
            num_class_as_given=num_class_as_given,
            beam_width=beam_width,
            log_learning_process=log_learning_process,
            log_folder_name=log_folder_name,
            dataset_name=dataset_name,
            feature_names=feature_names,
            label_names=label_names,
            which_features=which_features,
            random_seed=random_seed,
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
            "ModelEncoding.*", 
            "DataEncoding.*", 
            "DataInfo.*", 
            "utils_calculating_cl.*",  
            "exp_predictive_perf.*",
            "run_uci.*",
        ]

    def fit(self, X_train, y_train):

        # manager = MyManager()
        # manager.start()

        # data_info = manager.DataInfo(X=X_train, y=y_train, beam_width=None, alg_config=self.alg_config)

        data_info = DataInfo.DataInfo(X=X_train, y=y_train, beam_width=None, alg_config=self.alg_config)
        data_encoding = DataEncoding.NMLencoding(data_info)
        model_encoding = ModelEncoding.ModelEncodingDependingOnData(data_info)

        self.ruleset = Ruleset.Ruleset(
            data_info=data_info, 
            data_encoding=data_encoding, 
            model_encoding=model_encoding
        )

        self.ruleset.fit(max_iter=1000)
    
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

