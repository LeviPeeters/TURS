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
        
        # Question: why is this here
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
        """ Calculate the code length of a rule depending on data.
        Encodes the features one by one, discarding cutting points that become irrelevant as literals are "transmitted".
        Assumes that we can use the data to encode the model, i.e. the model is not needed to decode the data. 
        
        Parameters
        ---
        condition_matrix : Array
            Matrix of conditions for each literal in the rule
        col_orders : Array
            Contains indices of the features this rule uses, in the order they were added during growing

        Returns
        ---
        : float
            Code length of the rule
        """
        # Count the conditions on each feature. Can be 0, 1 or 2. 
        condition_count = (~np.isnan(condition_matrix[0])).astype(int) + (~np.isnan(condition_matrix[1])).astype(int)

        num_variables = np.count_nonzero(condition_count)
        l_num_variables = self.cached_cl_model["l_number_of_variables"][num_variables]
        l_which_variables = self.cached_cl_model["l_which_variables"][num_variables]

        bool_ = np.ones(len(self.data_info.features), dtype=bool)
        l_cuts = 0
        for index, col in enumerate(col_orders):
            # Calculate the bounds of this feature for the remaining data
            up_bound, low_bound = np.max(self.data_info.features[bool_, col]), np.min(self.data_info.features[bool_, col])
            num_cuts = np.count_nonzero((self.data_info.candidate_cuts[col] >= low_bound) &
                                        (self.data_info.candidate_cuts[col] <= up_bound))  # only an approximation here.
            
            # Code lenth required to encode the cutting point(s)
            if condition_count[col] == 1:
                if num_cuts == 0:
                    l_cuts += 0
                else:
                    l_cuts += np.log2(num_cuts)
            else:
                if num_cuts >= 2:
                    l_cuts += np.log2(num_cuts) + np.log2(num_cuts - 1) - np.log2(2)
                else:
                    l_cuts += 0

            # Now that we have "transmitted" a literal, we can drop data that the rule no longer covers
            # This make the code length of the next literal smaller, as some cutting points might not be relevant anymore
            if index != len(col_orders) - 1:
                assert condition_count[col] == 1 or condition_count[col] == 2
                if condition_count[col] == 1:
                    if not np.isnan(condition_matrix[0, col]):
                        bool_ = bool_ & (self.data_info.features[:, col] <= condition_matrix[0, col])
                    else:
                        bool_ = bool_ & (self.data_info.features[:, col] > condition_matrix[1, col])
                else:
                    bool_ = bool_ & ((self.data_info.features[:, col] <= condition_matrix[0, col]) &
                                     (self.data_info.features[:, col] > condition_matrix[1, col]))

        return l_num_variables + l_which_variables + l_cuts

    def cl_model_after_growing_rule(self, rule, ruleset, icol, cut_option):
        """ Calculate model code length after growing on a rule.
        If icol and cut_option are not None, the rule is still being grown.
        
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
            Code length of the ruleset after growing on the rule
        """
        
        # This should never happen, but TURS2 has code to catch this occurence so I'll check for it
        assert rule is not None, "Rule is None when computing model code length after growing"

        condition_count = np.array(rule.condition_count)
        icols_in_order = rule.icols_in_order
        condition_matrix = np.array(rule.condition_matrix)

        growing_rule = 0

        # when the rule is still being grown by adding condition using icol, we need to update the condition_count;
        if icol is not None and cut_option is not None:
            if np.isnan(rule.condition_matrix[cut_option, icol]):
                condition_count[icol] += 1
                # Note that this is just a place holder, to make this position not equal to np.nan; Need to make this more readable later.
                condition_matrix[0, icol] = np.inf

            if icol not in icols_in_order:
                icols_in_order = icols_in_order + [icol]
            
            growing_rule = 1

        # note that here is a choice based on the assumption that we can use $X$ to encode the model;
        cl_model_rule_after_growing = self.rule_cl_model_dep(condition_matrix, icols_in_order)

        # If we are growing a rule, this rule is not in the ruleset yet so we add 1 
        l_num_rules = utils_modelencoding.universal_code_integers(len(ruleset.rules) + growing_rule)
        
        # Cover redunancy in the order of rules
        cl_redundancy_rule_orders = math.lgamma(len(ruleset.rules) + 1 + growing_rule) / np.log(2)

        return l_num_rules + cl_model_rule_after_growing - cl_redundancy_rule_orders + ruleset.allrules_cl_model