# Class DataInfo which stores metadata about the dataset
# Lincen's code includes a number of methods to get candidate cuts, but only this one is used

import numpy as np
import os
from datetime import datetime
import logging
from scipy.sparse import csc_array

import utils_namedtuple
import utils_calculating_cl

class DataInfo:
    def __init__(self, X, y, beam_width=None, alg_config=None, not_use_excl_=None):
        if alg_config is None:
            assert beam_width is not None
            self.alg_config = utils_namedtuple.AlgConfig(
                num_candidate_cuts=100, max_num_rules=500, max_grow_iter=200, num_class_as_given=None,
                beam_width=beam_width,
                log_learning_process=False,
                dataset_name=None,
                feature_names=["X" + str(i) for i in range(X.shape[1])],
                label_names=np.unique(y),
                validity_check="either"
            )
        else:
            self.alg_config = alg_config

        self.not_use_excl_ = not_use_excl_

        # # Make sure X and y are numpy arrays
        # if type(X) != np.ndarray:
        #     self.features = X.to_numpy()
        # else:
        #     self.features = X
        # self.features = csr_array(self.features)

        # if type(y) != np.ndarray:
        #     self.target = y.to_numpy().flatten()
        # else:
        #     self.target = y

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
        
        # Set up a logging file 
        if self.alg_config.log_learning_process:
            time = datetime.now().strftime("%Y-%m-%d_%H-%M")
            if self.alg_config.log_folder_name:
                os.mkdir(f"logs/{self.alg_config.log_folder_name}")
                logging.basicConfig(filename=f"logs/{self.alg_config.log_folder_name}/log.txt", encoding='utf-8', level=logging.INFO, format='%(asctime)s %(message)s')
            else:
                os.mkdir(f"logs/{time}")
                logging.basicConfig(filename=f"logs/{time}/log.txt", encoding='utf-8', level=logging.DEBUG, format='%(asctime)s %(message)s')

            # Write the configuration to the log file
            logging.info(f"Log for learning process at {time}\n")
            logging.info(f"Algorithm Configuration:")
            for key, value in self.alg_config._asdict().items():
                if key != "feature_names":
                    logging.info(f"{key}: {value}")
            logging.info("\n")

    
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
