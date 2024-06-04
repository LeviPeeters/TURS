# Class DataInfo which stores metadata about the dataset
# Lincen's code includes a number of methods to get candidate cuts, but only this one is used

import numpy as np
import os
from datetime import datetime
import logging
from scipy.sparse import csc_array
import time

import utils_namedtuple
import utils_calculating_cl
import utils_modelencoding
import utils

class DataInfo:
    def __init__(self, X, y, beam_width=None, alg_config=None, not_use_excl_=None):
        if alg_config is None:
            assert beam_width is not None
            self.alg_config = utils_namedtuple.AlgConfig(
                num_candidate_cuts=100, 
                max_num_rules=500, 
                max_grow_iter=200, 
                num_class_as_given=None,
                beam_width=beam_width,
                chunksize=1,
                workers=-1,
                log_learning_process=1,
                dataset_name=None,
                feature_names=["X" + str(i) for i in range(X.shape[1])],
                label_names=np.unique(y),
                validity_check="either"
            )
        else:
            self.alg_config = alg_config

        self.not_use_excl_ = not_use_excl_

        # TODO: Currently, TURS only works with sparse matrices. Eventually, it should be possible to choose between sparse and dense matrices
        self.features = csc_array(X) # Sparse matrix
        self.target = y

        assert type(self.target) == np.ndarray

        # Parameters to be stored internally
        self.max_grow_iter = self.alg_config.max_grow_iter
        self.feature_names = self.alg_config.feature_names
        self.beam_width = self.alg_config.beam_width
        self.dataset_name = self.alg_config.dataset_name
        self.num_candidate_cuts = self.alg_config.num_candidate_cuts
        self.nrow, self.ncol = X.shape[0], X.shape[1]
        self.log_learning_process = self.alg_config.log_learning_process
        self.start_time = time.time()

        # Make a dictionary of categorical features and their possible values
        self.categorical_features = {}
        for name in self.feature_names:
            try:
                feature, value = name.split("_")
            except ValueError:
                pass
            else:
                if feature not in self.categorical_features:
                    self.categorical_features[feature] = [value]
                else:
                    self.categorical_features[feature].append(value)

        # get num_class, ncol, nrow,
        self.num_class = len(np.unique(self.target))
        if self.alg_config.num_class_as_given is not None:
            self.num_class = self.alg_config.num_class_as_given

        # Set the prior probabilities for the target    
        self.default_p = utils_calculating_cl.calc_probs(self.target, self.num_class)

        # Get candidate cut points for each numerical feature
        self.candidate_cuts = self.candidate_cuts_quantile_midpoints(self.num_candidate_cuts)

        # TODO: what does this do
        self.cached_number_of_rules_for_cl_model = self.alg_config.max_grow_iter

        self.cached_cl_model = {}
        self.max_num_rules = 100  # an upper bound for the number of rules, just for cl_model caching
        self.data_ncol_for_encoding = self.ncol
        self.cache_cl_model(self.ncol, self.max_grow_iter, self.candidate_cuts)
        
        # Set up logging files
        if self.alg_config.log_learning_process > 0:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
            if self.alg_config.log_folder_name:
                os.mkdir(f"logs/{self.alg_config.log_folder_name}")
                filename=f"logs/{self.alg_config.log_folder_name}/log"
            else:
                os.mkdir(f"logs/{timestamp}")
                filename=f"logs/{timestamp}/log"

            # Set up a text logger
            handler = logging.FileHandler(filename=filename+".txt", encoding='utf-8', mode='w')
            # handler.setFormatter(utils.ElapsedTimeFormatter())
            handler.setFormatter(utils.RegularFormatter())
            self.logger: logging.RootLogger
            self.logger = logging.getLogger("text_logger")
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

            # Write the configuration to the log file
            self.logger.info(f"Log for learning process at {timestamp}\n")
            self.logger.info(f"Algorithm Configuration:")
            for key, value in self.alg_config._asdict().items():
                if key != "feature_names":
                    self.logger.info(f"{key}: {value}")
            self.logger.info("Number of features: " + str(self.ncol))
            self.logger.info("\n")

            # For logging information about the growth process to a CSV file
            handler3 = logging.FileHandler(filename=filename+"_growth.csv", encoding='utf-8', mode='w')
            handler3.setFormatter(logging.Formatter('%(message)s'))
            self.growth_logger: logging.RootLogger
            self.growth_logger = logging.getLogger("growth_logger")
            self.growth_logger.addHandler(handler3)
            self.growth_logger.setLevel(logging.INFO)
            self.growth_logger.info("rule,iteration,coverage_incl,coverage_excl,mdl_gain_incl,mdl_gain_excl")

            self.current_rule = 0
            self.current_iteration = 0

            if self.log_learning_process > 2:
                # For logging the time taken for each function to a CSV file
                handler2 = logging.FileHandler(filename=filename+"_time.csv", encoding='utf-8', mode='w')
                handler2.setFormatter(logging.Formatter('%(message)s'))
                self.time_logger: logging.RootLogger
                self.time_logger = logging.getLogger("time_logger")
                self.time_logger.addHandler(handler2)
                self.time_logger.setLevel(logging.INFO)
                self.time_logger.info("Thread,Time,Function")
    
    def cache_cl_model(self, data_ncol, max_rule_length, candidate_cuts):
        """ Precompute certain code length values we'll need often
        For number of variables, number of rules, number of cuts and selection of variables

        Parameters
        ---
        data_ncol : int
            Number of variables
        max_rule_length : int
            Maximum number of literals in a rule
        candidate_cuts : Dict
            Candidate cut points for each variable
        
        Returns
        ---
        None
        
        """ 
        # Code Length contributed by the number of variables in a rule (one per literal)
        l_number_of_variables = [utils_modelencoding.universal_code_integers(i) for i in range(max(data_ncol-1, max_rule_length))]
        
        # Code Length contributed to specify which variables are used in a rule
        l_which_variables = utils_modelencoding.log2comb(data_ncol, np.arange(max(data_ncol-1, max_rule_length)))

        # Code Length contributed by the number of rules in the ruleset
        l_number_of_rules = [utils_modelencoding.universal_code_integers(i) for i in range(self.max_num_rules)]

        # Code length contributed by a cut is dependent on the number of candidate cuts, so make an array of those
        candidate_cuts_length = np.array([len(candi) for candi in candidate_cuts.values()], dtype=float)
        only_one_candi_selector = (candidate_cuts_length == 1)
        zero_candi_selector = (candidate_cuts_length == 0)

        # Code length contributed by the cut point for a literal, if there is one cut
        l_one_cut = np.zeros(len(candidate_cuts_length), dtype=float)
        l_one_cut[~zero_candi_selector] = np.log2(candidate_cuts_length[~zero_candi_selector]) + 1 + 1  # 1 bit for LEFT/RIGHT, and 1 bit for one/two cuts
        l_one_cut[only_one_candi_selector] = l_one_cut[only_one_candi_selector] - 1

        # Code length contributed by the cut point for a literal, if there are two cuts (interval)
        l_two_cut = np.zeros(len(candidate_cuts_length))
        l_two_cut[only_one_candi_selector] = np.nan
        two_candi_selector = (candidate_cuts_length > 1)

        # TODO: reconsider the l_two_cut from the perspective of hypothesis testing
        l_two_cut[two_candi_selector] = np.log2(candidate_cuts_length[two_candi_selector]) + \
                                        np.log2(candidate_cuts_length[two_candi_selector] - 1) - np.log2(2) \
                                        + 1  # the last 1 bit is for encoding one/two cuts
        
        # Store precomputed values
        l_cut = np.array([l_one_cut, l_two_cut])
        self.cached_cl_model["l_number_of_variables"] = l_number_of_variables
        self.cached_cl_model["l_cut"] = l_cut
        self.cached_cl_model["l_which_variables"] = l_which_variables
        self.cached_cl_model["l_number_of_rules"] = l_number_of_rules

    def candidate_cuts_quantile_midpoints(self, num_candidate_cuts):
        """ Calculate the candidate cuts for each numerical feature, using the quantile midpoints method.
        
        Parameters
        ---
        num_candidate_cuts : list or int
            Number of candidate cuts to use
            If this is a list, each feature's number of cuts is taken from the list
            If this is an int, that number is used for all features
        
        Returns
        ---
        candidate_cuts : Dict
            Dictionary of candidate cuts for each feature
        """
        candidate_cuts = {}

        if type(num_candidate_cuts) is not list:
            num_candidate_cuts = [num_candidate_cuts] * self.ncol
        
        for i, feature in enumerate(self.features.T):
            # We make the feature dense to avoid issues with sparse matrix indexing
            unique_feature = np.unique(feature.todense())
            # print(unique_feature)
            # breakpoint()
            if len(unique_feature) <= 1:
                # This can happen because of cross-validation
                candidate_cut_this_dimension = np.array([], dtype=float)
                candidate_cuts[i] = candidate_cut_this_dimension
            elif np.array_equal(unique_feature, np.array([0, 1])):
                # Binary feature only has one cut
                candidate_cuts[i] = np.array([0.5])
            else:
                # Sort the list and ensure no duplicate values. Duplicates can cause the quantiles to not be the same size
                sort_feature = np.sort(feature.todense()+np.random.uniform(0, 1e-10, feature.shape[0])).flatten()

                # Calculate the midpoints between consecutive values
                midpoints = (sort_feature[:-1] + sort_feature[1:]) / 2

                # We don't want more cut points than there are unique values
                num_candidate_cuts_i = np.min([len(unique_feature) - 1, num_candidate_cuts[i]])

                # Now select the midpoints to use
                if (num_candidate_cuts[i] > 1) & (len(midpoints) > num_candidate_cuts_i):
                    select_indices = np.linspace(0, len(midpoints) - 1, num_candidate_cuts_i + 2,
                                                 endpoint=True, dtype=int)
                    select_indices = select_indices[1:(len(select_indices) - 1)]  # remove the start and end point
                    candidate_cuts[i] = midpoints[select_indices]
                else:
                    candidate_cuts[i] = midpoints
        return candidate_cuts
