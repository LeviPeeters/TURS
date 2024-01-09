import numpy as np
import math

import utils_modelencoding
import DataInfo

class ModelEncodingDependingOnData:
    def __init__(self, 
                 data_info: DataInfo.DataInfo, 
                 given_ncol=None):
        self.cached_cl_model = {}
        self.max_num_rules = 100  # an upper bound for the number of rules, just for cl_model caching
        self.data_info = data_info
        if given_ncol is None:
            self.data_ncol_for_encoding = data_info.ncol
            self.cache_cl_model(data_info.ncol, data_info.max_grow_iter, data_info.candidate_cuts)
        else:
            self.data_ncol_for_encoding = given_ncol
            self.cache_cl_model(given_ncol, data_info.max_grow_iter, data_info.candidate_cuts)

    def cache_cl_model(self, data_ncol, max_rule_length, candidate_cuts):
        """ Precompute certain code length values we'll need often
        For number of variables, number of rules, number of cuts and selection of variables
        TODO: review. I'm not sure about this description. 

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
        
        # TODO: why is this here
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
    
    def rule_cl_model(self, condition_count):
        """ Retrieve the code length of a rule using the precomputed values in the cache.
        TODO: Remove, because it's not actually used. Keep it for now because it helps my understanding

        Parameters
        ---
        condition_count : Array
            Number of conditions (1 or 2) for each literal in the rule
        
        Returns
        ---
        : float
            Code length of the rule
        """
        num_variables = np.count_nonzero(condition_count)
        l_num_variables = self.cached_cl_model["l_number_of_variables"][num_variables]
        l_which_variables = self.cached_cl_model["l_which_variables"][num_variables]
        l_cuts = (
                np.sum(self.cached_cl_model["l_cut"][0][condition_count == 1]) +
                np.sum(self.cached_cl_model["l_cut"][1][condition_count == 2])
        )

        return l_num_variables + l_which_variables + l_cuts
    
    def rule_cl_model_dep(self, condition_matrix, col_orders):
        """ Calculate the code length of a rule, dependent on the data
        TODO: Revieuw. I'm not sure about this description
        
        Parameters
        ---
        condition_matrix : Array
            Matrix of conditions for each literal in the rule
        col_orders : Array
            Order of the columns in the data

        Returns
        ---
        : float
            Code length of the rule
        """
        pass

    def cl_model_after_growing_on_rule(self, rule, ruleset, icol, cut_option):
        """ TODO

        Parameters
        ---
        rule : Rule object
            Rule under consideration
        ruleset : Ruleset object
            Full ruleset 
        icol : int
            Index of the feature under consideration
        cut_option : int
            Cut point under consideration
        
        Returns
        ---
        : float
            Code length of the data and ruleset after growing on the rule
        """
        pass