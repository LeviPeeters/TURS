import numpy as np
import sys
import dill
import weakref
import logging
from functools import partial
import tqdm

import utils_calculating_cl
import nml_regret
import Beam
import DataInfo
import time
from scipy.sparse import csc_array
import multiprocessing as mp
import math

import constant
import utils_modelencoding
from utils_namedtuple import RulesetInfo

def store_grow_info(excl_bi_array, incl_bi_array, icol, cut, cut_option, excl_mdl_gain, incl_mdl_gain,
                    coverage_excl, coverage_incl, normalized_gain_excl, normalized_gain_incl):
    """ Store the information of a grow step in a dictionary."""
    try:
        excl_coverage = np.count_nonzero(excl_bi_array)
        incl_coverage = np.count_nonzero(incl_bi_array)
        
        # locals() returns a dictionary containing all local variables
        return locals()
    
    except:
        logging.error("Error in store_grow_info")
        raise
    
def store_grow_info_rulelist(excl_bi_array, icol, cut, cut_option, excl_normalized_gain):
    return {"excl_bi_array": excl_bi_array, "icol": icol, "cut": cut,
            "cut_option": cut_option, "excl_normalized_gain": excl_normalized_gain}

def setup_worker(data_info_orig, ruleset_info_orig, modelling_groups_orig):
    global data_info
    data_info = data_info_orig

    global ruleset_info 
    ruleset_info = ruleset_info_orig

    global modelling_groups
    modelling_groups = modelling_groups_orig

def worker_wrapper(worker_args, literal_args):
    worker = RuleWorker(*worker_args)
    return worker.grow_one_literal(*literal_args)

class Rule:
    def __init__(self, 
                 indices, 
                 indices_excl, 
                 data_info: DataInfo.DataInfo,
                 rule_base, 
                 condition_matrix, 
                 ruleset,
                 mdl_gain, 
                 mdl_gain_excl, 
                 icols_in_order):  

        self.ruleset = ruleset 
        self.data_info = data_info # Metadata
        self.rule_base = rule_base  # The previous level of this rule
        self.icols_in_order = icols_in_order  # The order of the columns in the 
        
        # Needed for data encoding
        self.num_class = data_info.num_class
        self.calc_probs = partial(utils_calculating_cl.calc_probs, num_class=self.num_class)

        # Numpy arrays with indices of covered indices
        self.indices = indices  
        self.indices_excl = indices_excl  

        # Boolean arrays indicating covered indices
        self.bool_array = self.get_bool_array(self.indices)
        self.bool_array_excl = self.get_bool_array(self.indices_excl)

        # Number of instances covered by this rule
        self.coverage = len(self.indices) 
        self.coverage_excl = len(self.indices_excl)

        # The feature arrays take up far too much memory, especially because empty rules cover the entire dataset\
        self.target = data_info.target[indices]
        self.target_excl = data_info.target[indices_excl]

        # Condition matrix containing the rule literals and a boolean array to show which features have a condition
        self.condition_matrix = condition_matrix
        self.condition_bool = (~np.isnan(condition_matrix[0])).astype(int) + (~np.isnan(condition_matrix[1])).astype(int)

        # Probabilities and regrets
        self.prob_excl = self._calc_probs(target=self.target_excl)
        self.prob = self._calc_probs(target=self.target)
        self.regret_excl = nml_regret.regret(len(self.indices_excl), data_info.num_class)
        self.regret = nml_regret.regret(len(self.indices), data_info.num_class)
        self.negloglike_excl = utils_calculating_cl.calc_negloglike(p=self.prob_excl, n=len(self.indices_excl))
        self.cl_model = self.rule_cl_model_dep(self.condition_matrix, col_orders=icols_in_order)

        self.mdl_gain = mdl_gain
        self.mdl_gain_excl = mdl_gain_excl

        if self.rule_base is None:
            self.incl_mdl_gain, self.excl_mdl_gain = -np.Inf, -np.Inf
            self.incl_gain_per_excl_coverage, self.excl_gain_per_excl_coverage = -np.Inf, -np.Inf
        else:
            self.incl_mdl_gain, self.excl_mdl_gain = mdl_gain, mdl_gain_excl
            if self.coverage_excl == 0:
                self.incl_gain_per_excl_coverage, self.excl_gain_per_excl_coverage = np.nan, np.nan
            else:
                self.incl_gain_per_excl_coverage, self.excl_gain_per_excl_coverage = mdl_gain / self.coverage_excl, mdl_gain_excl / self.coverage_excl

    def _calc_probs(self, target):
        """ Given a vector containing the targets of a set of instances, calculate the probability of encountering each target.

        Parameters
        ---
        target : Array
            Vector containing the targets of a set of instances
        
        Returns
        ---
        : Array
            Vector containing the probability of encountering each target
        """
        return utils_calculating_cl.calc_probs(target, num_class=self.data_info.num_class)

    def get_bool_array(self, indices):
        """ Given a list of indices, return a boolean array with True at the indices in the list.
        
        Parameters
        ---
        indices : list
            
        Returns
        ---
        bool_array : Array
            Boolean array with True at the indices in the list
        """
        bool_array = np.zeros(self.data_info.nrow, dtype=bool)
        bool_array[indices] = True
        return bool_array

    def get_candidate_cuts_icol_given_rule(self, candidate_cuts, icol):
        """ Given a list of candidate cuts and an index of a feature, return the candidate cuts for that feature.
        Only returns candidate cuts that have remaining data points on both sides of the cut.
        
        Parameters
        ---
        candidate_cuts : Array
            List of candidate cuts
        icol : int
            Index of the feature for which the candidate cuts are being considered
            
        Returns
        candidate_cuts_icol : Array
            List of candidate cuts for the feature 
        """
        features = self.data_info.features
        if self.rule_base is None:
            candidate_cuts_selector = (candidate_cuts[icol] < np.max(features[self.indices_excl, [icol]].flatten())) & \
                                      (candidate_cuts[icol] > np.min(features[self.indices_excl, [icol]].flatten()))
            candidate_cuts_icol = candidate_cuts[icol][candidate_cuts_selector]
        else:
            candidate_cuts_selector = (candidate_cuts[icol] < np.max(features[self.indices, [icol]])) & \
                                      (candidate_cuts[icol] > np.min(features[self.indices, [icol]]))
            candidate_cuts_icol = candidate_cuts[icol][candidate_cuts_selector]
        return candidate_cuts_icol

    def grow(self, incl_or_excl, log=False):
        if self.data_info.alg_config.log_learning_process > 0 and log:
            self.data_info.growth_logger.info(f"{self.data_info.current_rule},{self.data_info.current_iteration},{self.coverage},{self.coverage_excl},{self.mdl_gain},{self.mdl_gain_excl}")
        if self.data_info.alg_config.log_learning_process > 1 and log:    
            self.data_info.logger.info(str(self))

        s = time.time()

        beam = Beam.DiverseCovBeam(width=self.data_info.beam_width)
        
        candidate_cuts = self.data_info.candidate_cuts

        # Make a list of all literals to be grown
        literals = []
        for icol in range(self.data_info.ncol):
            bi_array = self.data_info.features[:, [icol]].todense().flatten()
            candidate_cuts_icol = self.get_candidate_cuts_icol_given_rule(candidate_cuts, icol)

            for cut in candidate_cuts_icol:
                valid = True

                 # Construct binary arrays indicating which features fall on each side of the cut
                excl_left_bi_array = (bi_array[self.indices_excl] < cut)
                excl_right_bi_array = ~excl_left_bi_array
                left_bi_array = (bi_array[self.indices] < cut)
                right_bi_array = ~left_bi_array

                # Store all of the binary arrays in a dictionary to make them easy to pass to validity check
                bi_arrays = {"left": left_bi_array, 
                                "right": right_bi_array, 
                                "excl_left": excl_left_bi_array,
                                "excl_right": excl_right_bi_array}

                # Check validity and skip if not valid
                _validity = self.validity_check(icol=icol, bi_arrays=bi_arrays)

                if self.data_info.not_use_excl_:
                    _validity["res_excl"] = False

                if _validity["res_excl"] == False and _validity["res_incl"] == False:
                    valid = False 

                excl_left_coverage, excl_right_coverage = np.count_nonzero(excl_left_bi_array), np.count_nonzero(
                    excl_right_bi_array)

                # Question: Why is there no check on incl_coverage being 0?
                if excl_left_coverage == 0 or excl_right_coverage == 0:
                    valid = False 
                
                if valid:
                    literals.append((icol, cut))

        # It's possible that no literals are valid, in which case we return the empty beam
        if len(literals) == 0:
            return beam

        # This information is needed in the growth process
        # The pool setup makes it globally available, so we don't need to pass the whole ruleset to the worker
        ruleset_info = RulesetInfo( total_cl=self.ruleset.total_cl, 
                                    n_rules=len(self.ruleset.rules), 
                                    allrules_cl_model=self.ruleset.allrules_cl_model,
                                    uncovered_bool=self.ruleset.uncovered_bool,
                                    allrules_cl_data=self.ruleset.allrules_cl_data,
                                    uncovered_indices=self.ruleset.uncovered_indices,
                                    allrules_regret=self.ruleset.allrules_regret)

        # TODO: I think it's possible to keep the pool around for following iterations 
        if self.data_info.alg_config.workers == -1:
            pool = mp.Pool(mp.cpu_count(), initializer=setup_worker, initargs=(self.data_info, ruleset_info, self.ruleset.modelling_groups))
        else:
            pool = mp.Pool(self.data_info.alg_config.workers, initializer=setup_worker, initargs=(self.data_info, ruleset_info, self.ruleset.modelling_groups))


        # Define the global variables that the workers need
        # setup_worker(self.data_info, ruleset_info, self.ruleset.modelling_groups)

        # I do this here so rule_base is not passed to the worker and we don't have to worry about pickling it
        if self.rule_base is None:
            incl_mdl_gain, excl_mdl_gain = -np.Inf, -np.Inf
            incl_gain_per_excl_coverage, excl_gain_per_excl_coverage = -np.Inf, -np.Inf
        else:
            incl_mdl_gain, excl_mdl_gain = self.mdl_gain, self.mdl_gain_excl
            if self.coverage_excl == 0:
                incl_gain_per_excl_coverage, excl_gain_per_excl_coverage = np.nan, np.nan
            else:
                incl_gain_per_excl_coverage, excl_gain_per_excl_coverage = self.mdl_gain / self.coverage_excl, self.mdl_gain_excl / self.coverage_excl
        
        # Initialize the worker class
        # TODO no way these are all needed
        worker_args = (self.indices, 
                            self.indices_excl, 
                            self.condition_matrix, 
                            self.mdl_gain, 
                            self.mdl_gain_excl, 
                            self.icols_in_order,
                            incl_mdl_gain,
                            excl_mdl_gain,
                            incl_gain_per_excl_coverage,
                            excl_gain_per_excl_coverage)
        
        # with dill.detect.trace("dill_trace.log", mode='w'):
        #     dill.dumps(worker)

        results = mp.Manager().list([None] * len(literals))

        # If chunksize is 97, perform experiment with varying chunksize
        if self.data_info.chunksize == 97:
            chunksize = len(literals)//mp.cpu_count()
            if chunksize < 1:
                chunksize = 1
            else:
                chunksize = math.floor(chunksize)
        else:
            chunksize = self.data_info.chunksize

        res = pool.starmap_async(worker_wrapper, tqdm.tqdm([(worker_args, (incl_or_excl, literal[0], literal[1], results, i, log)) for i, literal in enumerate(literals)], total=len(literals)), chunksize=chunksize)
        res.wait()
        pool.close()

        # setup_worker(self.data_info, ruleset_info, self.ruleset.modelling_groups)
        # worker = RuleWorker(*worker_args)
        # [worker.grow_one_literal(incl_or_excl, literal[0], literal[1], i, log) for i, literal in enumerate(literals)]

        if self.data_info.alg_config.log_learning_process > 0 and log:
            n_literals = len([1 for result in results if result is not None])
            self.data_info.literal_logger.info(f"{n_literals},{(time.time()-s)/n_literals}")

        for result in results:

            if result is None:
                continue
            else:
                left_grow_info, right_grow_info = result

            left_grow_info["_rule"] = self
            right_grow_info["_rule"] = self
            beam.update(left_grow_info, left_grow_info[f"normalized_gain_{incl_or_excl}"])
            beam.update(right_grow_info, right_grow_info[f"normalized_gain_{incl_or_excl}"])
        return beam

    def rule_cl_model_dep(self, condition_matrix, col_orders, log=False):
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
        try:
            if self.data_info.log_learning_process > 2 and log:
                s = time.time()
            # Count the conditions on each feature. Can be 0, 1 or 2. 
            condition_count = (~np.isnan(condition_matrix[0])).astype(int) + (~np.isnan(condition_matrix[1])).astype(int)

            num_variables = np.count_nonzero(condition_count)
            l_num_variables = self.data_info.cached_cl_model["l_number_of_variables"][num_variables]
            l_which_variables = self.data_info.cached_cl_model["l_which_variables"][num_variables]

            bool_ = np.ones(self.data_info.features.shape[0], dtype=bool)
            covered_features = self.data_info.features[bool_, :]

            l_cuts = 0

            for index, col in enumerate(col_orders):
                s = time.time()
                feature = covered_features[:, [col]]
                up_bound, low_bound = np.max(feature), np.min(feature)

                s = time.time()
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

                s = time.time()
                # Now that we have "transmitted" a literal, we can drop data that the rule no longer covers
                # This make the code length of the next literal smaller, as some cutting points might not be relevant anymore
                if index != len(col_orders) - 1:
                    assert condition_count[col] == 1 or condition_count[col] == 2
                    if condition_count[col] == 1:
                        if not np.isnan(condition_matrix[0, col]):
                            bool_ = bool_ & (self.data_info.features[:, [col]].todense().flatten() <= condition_matrix[0, col])
                        else:
                            bool_ = bool_ & (self.data_info.features[:, [col]].todense().flatten() > condition_matrix[1, col])
                    else:
                        temp = self.data_info.features[:, [col]].todense().flatten()
                        bool_ = bool_ & ((temp <= condition_matrix[0, col]) &
                                        (temp > condition_matrix[1, col]))

            return l_num_variables + l_which_variables + l_cuts
        except:
            self.data_info.logger.error("Error in rule_cl_model_dep")
            raise

    def cl_model_after_growing_rule(self, ruleset, icol, cut_option, log=False):
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
        try:
            condition_count = np.array(self.condition_bool)
            icols_in_order = self.icols_in_order
            condition_matrix = np.array(self.condition_matrix)

            growing_rule = 0

            # when the rule is still being grown by adding condition using icol, we need to update the condition_count;
            if icol is not None and cut_option is not None:
                if np.isnan(self.condition_matrix[cut_option, icol]):
                    condition_count[icol] += 1
                    # Note that this is just a place holder, to make this position not equal to np.nan; Need to make this more readable later.
                    condition_matrix[0, icol] = np.inf

                if icol not in icols_in_order:
                    icols_in_order = icols_in_order + [icol]
                
                growing_rule = 1

            # note that here is a choice based on the assumption that we can use $X$ to encode the model;
            cl_model_rule_after_growing = self.rule_cl_model_dep(condition_matrix, icols_in_order, log=log)

            # If we are growing a rule, this rule is not in the ruleset yet so we add 1 
            l_num_rules = utils_modelencoding.universal_code_integers(len(ruleset.rules) + growing_rule)
            
            # Cover redunancy in the order of rules
            cl_redundancy_rule_orders = math.lgamma(len(ruleset.rules) + 1 + growing_rule) / np.log(2)

            return l_num_rules + cl_model_rule_after_growing - cl_redundancy_rule_orders + ruleset.allrules_cl_model
        except:
            self.data_info.logger.error("Error in cl_model_after_growing_rule")
            raise

    def validity_check(self, icol, bi_arrays):
        """ Control function for validity check

        
        Parameters
        ---
        rule : Rule object
            Rule to be checked
        icol : int
            Index of the column for which the candidate cut is calculated
        cut : float
            Candidate cut
        bi_arrays : dict
            Binary arrays containing the instances covered by the rule left and right of the cut
        
        Returns
        ---
        : dict
            contains the validity including and excluding overlapping instances
        """
        try:
            res_excl = True
            res_incl = True
            if self.data_info.alg_config.validity_check == "no_check":
                pass
            elif self.data_info.alg_config.validity_check == "excl_check":
                res_excl = self.check_split_validity_excl(icol, bi_arrays)
            elif self.data_info.alg_config.validity_check == "incl_check":
                res_incl = self.check_split_validity(icol, bi_arrays)
            elif self.data_info.alg_config.validity_check == "either":
                res_excl = self.check_split_validity_excl(icol, bi_arrays)
                res_incl = self.check_split_validity(icol, bi_arrays)
            else:
                # TODO: I should improve this error message a bit
                sys.exit("Error: the if-else statement should not end up here")
            return {"res_excl": res_excl, "res_incl": res_incl}
        except:
            self.data_info.logger.error(f"Error in validity_check:")
            raise

    def check_split_validity(self, icol, bi_arrays):
        """ Check validity when considering overlapping instances
        
        Parameters
        ---
        rule : Rule object
            Rule to be checked
        icol : int
            Index of the column for which the candidate cut is calculated
        bi_arrays : dict
            Binary arrays containing the instances covered by the rule left and right of the cut
        
        Returns
        ---
        : bool
            Whether the split is valid
        """
        try:
            indices_left, indices_right = np.where(bi_arrays["left"])[0], np.where(bi_arrays["right"])[0]

            # Compute probabilities for the coverage of the rule and both sides of the split
            p_left = utils_calculating_cl.calc_probs(self.data_info.target[indices_left], self.data_info.num_class)
            p_right = utils_calculating_cl.calc_probs(self.data_info.target[indices_right], self.data_info.num_class)

            # Compute the negative log-likelihoods for these probabilities
            nll_rule = utils_calculating_cl.calc_negloglike(self.prob, self.coverage)
            nll_left = utils_calculating_cl.calc_negloglike(p_left, len(indices_left))
            nll_right = utils_calculating_cl.calc_negloglike(p_right, len(indices_right))

            # The extra model code length this split would add
            cl_model_extra = self.data_info.cached_cl_model["l_cut"][0][icol]  # 0 represents for the "one split cut", instead of "two-splits cut"
            cl_model_extra += np.log2(self.data_info.data_ncol_for_encoding)
            num_vars = np.sum(self.condition_bool > 0)
            cl_model_extra += self.data_info.cached_cl_model["l_number_of_variables"][num_vars + 1] - \
                            self.data_info.cached_cl_model["l_number_of_variables"][num_vars]

            # if this validity score is positive, the split decreases the total code length
            validity = nll_rule + nml_regret.regret(self.coverage, self.data_info.num_class) \
                        - nll_left - nml_regret.regret(len(indices_left), self.data_info.num_class) \
                        - nll_right - nml_regret.regret(len(indices_right), self.data_info.num_class) \
                        - cl_model_extra

            return (validity > 0)
        except:
            data_info.logger.error(f"Error in check_split_validity:")
            raise
        

    def check_split_validity_excl(self, icol, bi_arrays):
        """ Check validity without considering overlapping instances
        
        Parameters
        ---
        rule : Rule object
            Rule to be checked
        icol : int
            Index of the column for which the candidate cut is calculated
        bi_arrays : dict
            Binary arrays containing the instances covered by the rule left and right of the cut
        
        Returns
        ---
        : bool
            Whether the split is valid
        """
        try:
            indices_left, indices_right = np.where(bi_arrays["excl_left"])[0], np.where(bi_arrays["excl_right"])[0]

            # Compute probabilities for the coverage of the rule and both sides of the split
            p_left = utils_calculating_cl.calc_probs(self.data_info.target[indices_left], self.data_info.num_class)
            p_right = utils_calculating_cl.calc_probs(self.data_info.target[indices_right], self.data_info.num_class)

            # Compute the negative log-likelihoods for these probabilities
            nll_rule = utils_calculating_cl.calc_negloglike(self.prob_excl, self.coverage_excl)
            nll_left = utils_calculating_cl.calc_negloglike(p_left, len(indices_left))
            nll_right = utils_calculating_cl.calc_negloglike(p_right, len(indices_right))

            # The extra model code length this split would add
            cl_model_extra = self.data_info.cached_cl_model["l_cut"][0][icol]  # 0 represents for the "one split cut", instead of "two-splits cut"
            cl_model_extra += np.log2(self.data_info.data_ncol_for_encoding)
            num_vars = np.sum(self.condition_bool > 0)
            cl_model_extra += self.data_info.cached_cl_model["l_number_of_variables"][num_vars + 1] - \
                            self.data_info.cached_cl_model["l_number_of_variables"][num_vars]

            # if this validity score is positive, the split decreases the total code length
            validity = nll_rule + nml_regret.regret(self.coverage_excl, self.data_info.num_class) \
                        - nll_left - nml_regret.regret(len(indices_left), self.data_info.num_class) \
                        - nll_right - nml_regret.regret(len(indices_right), self.data_info.num_class) \
                        - cl_model_extra

            return (validity > 0)
        except:
            data_info.logger.error(f"Error in check_split_validity_excl:")
            raise

    def __str__(self):
        return self.to_string()

    def to_string(self, verbose=0):
        """ String representation of the rule for printing
        Should print categorical features in a more readable way
        
        Parameters
        ---
        verbose : bool
            If True, adds all labels and their probabilities at the end of the rule
            If false, only prints the most likely label and its probablity
            
        Returns
        ---
        readable : str
            String representation of the rule
        """
        feature_names = self.ruleset.data_info.feature_names
        label_names = self.ruleset.data_info.alg_config.label_names
        readable = "If: "
        which_variables = np.where(self.condition_bool != 0)[0]

        # We keep track of all negations, because we need to do some logic to print them correctly
        negations = {}

        if len(which_variables) == 0:
            return "Empty rule"

        for v in which_variables:
            cut = self.condition_matrix[:, v][::-1]
            icol_name = str(feature_names[v])

            # If we can split the feature name, it is categorical
            try:
                feature_name, value_name = icol_name.split("_")

            except ValueError:
                feature_name = icol_name
                value_name = None
            
            # For numerical features, we can just print the literal
            if value_name is None:
                if np.isnan(cut[0]):
                    readable += f"{icol_name} < {round(cut[1], 2)};    "
                elif np.isnan(cut[1]):
                    readable += f"{icol_name} >= {round(cut[0], 2)};    "
                else:
                    readable += f"{round(cut[0], 2)} <= {icol_name} < {round(cut[1], 2)};    "
            
            # For categorical features, we need to do a bit more work
            else:
                # if cut[1] is nan, it means the feature is 1, so the feature is equal to the value
                # Because one rule can't contain two values for one feature, we can print the literal as feature == value 
                if np.isnan(cut[1]):
                    readable += f"[b] {feature_name} == {value_name};    "
                
                # If cut[0] is nan, it means the feature is 0, so the feature is not equal to the value
                # In this case, we need to decide if it is more readable to print which values it CAN be, or which values it CAN'T be
                # To do so, we need to know all negations in this rule
                else:
                    if feature_name not in negations:
                        negations[feature_name] = [value_name]
                    else:
                        negations[feature_name].append(value_name)
       
        # Now we deal with the negations
        for feature, values in negations.items():
            if len(values) < len(self.ruleset.data_info.categorical_features[feature]) / 2:
                readable += f"{feature} != {', '.join(values)};    "
            else:
                readable += f"{feature} == {', '.join([v for v in self.ruleset.data_info.categorical_features[feature] if v not in values])};    "

        if verbose:
            readable += "\n"

            readable += "Then:\n"
            if len(self.prob) > 5:
                readable += f"Highest probability is {max(self.prob)} for outcome {label_names[np.argmax(self.prob)]}"
            else:
                for i in range(len(label_names)):
                    readable += f"Probability of {label_names[i]} is {round(self.prob[i], 2)}\n"
            readable += f"Coverage of this rule: {self.coverage}\n"
        else:
            readable += f"Then: {label_names[np.argmax(self.prob)]} (p={round(max(self.prob), 2)}, n={self.coverage})"

        return readable
    
class RuleWorker:
    def __init__(self, 
                 indices, 
                 indices_excl, 
                 condition_matrix, 
                 mdl_gain, 
                 mdl_gain_excl, 
                 icols_in_order,
                 incl_mdl_gain,
                 excl_mdl_gain,
                 incl_gain_per_excl_coverage,
                 excl_gain_per_excl_coverage):  

        # These are the global variables that the worker needs
        global data_info
        global ruleset_info
        global modelling_groups

        self.icols_in_order = icols_in_order  # The order of the columns in the 
        
        # Needed for data encoding
        self.num_class = data_info.num_class
        self.calc_probs = partial(utils_calculating_cl.calc_probs, num_class=self.num_class)

        # Numpy arrays with indices of covered indices
        self.indices = indices  
        self.indices_excl = indices_excl  

        # Boolean arrays indicating covered indices
        self.bool_array = self.get_bool_array(self.indices)
        self.bool_array_excl = self.get_bool_array(self.indices_excl)

        # Number of instances covered by this rule
        self.coverage = len(self.indices) 
        self.coverage_excl = len(self.indices_excl)

        # The feature arrays take up far too much memory, especially because empty rules cover the entire dataset\
        self.target = data_info.target[indices]
        self.target_excl = data_info.target[indices_excl]

        # Condition matrix containing the rule literals and a boolean array to show which features have a condition
        self.condition_matrix = condition_matrix
        self.condition_bool = (~np.isnan(condition_matrix[0])).astype(int) + (~np.isnan(condition_matrix[1])).astype(int)

        # Probabilities and regrets
        self.prob_excl = self._calc_probs(target=self.target_excl)
        self.prob = self._calc_probs(target=self.target)
        self.regret_excl = nml_regret.regret(len(self.indices_excl), data_info.num_class)
        self.regret = nml_regret.regret(len(self.indices), data_info.num_class)
        self.negloglike_excl = utils_calculating_cl.calc_negloglike(p=self.prob_excl, n=len(self.indices_excl))
        self.cl_model = self.rule_cl_model_dep(self.condition_matrix, col_orders=icols_in_order)

        self.mdl_gain = mdl_gain
        self.mdl_gain_excl = mdl_gain_excl
        self.incl_mdl_gain = incl_mdl_gain
        self.excl_mdl_gain = excl_mdl_gain
        self.incl_gain_per_excl_coverage = incl_gain_per_excl_coverage
        self.excl_gain_per_excl_coverage = excl_gain_per_excl_coverage


    def _calc_probs(self, target):
        """ Given a vector containing the targets of a set of instances, calculate the probability of encountering each target.

        Parameters
        ---
        target : Array
            Vector containing the targets of a set of instances
        
        Returns
        ---
        : Array
            Vector containing the probability of encountering each target
        """
        return utils_calculating_cl.calc_probs(target, num_class=data_info.num_class)

    def get_bool_array(self, indices):
        """ Given a list of indices, return a boolean array with True at the indices in the list.
        
        Parameters
        ---
        indices : list
            
        Returns
        ---
        bool_array : Array
            Boolean array with True at the indices in the list
        """
        bool_array = np.zeros(data_info.nrow, dtype=bool)
        bool_array[indices] = True
        return bool_array

    def grow_one_literal(self, incl_or_excl, icol, cut, result_list, result_index, log=False):
        try:
            s = time.time()
            bi_array = data_info.features[:, [icol]].todense().flatten()

            # Construct binary arrays indicating which features fall on each side of the cut
            excl_left_bi_array = (bi_array[self.indices_excl] < cut)
            excl_right_bi_array = ~excl_left_bi_array
            left_bi_array = (bi_array[self.indices] < cut)
            right_bi_array = ~left_bi_array

            # Store all of the binary arrays in a dictionary to make them easy to pass to validity check
            # bi_arrays = {"left": left_bi_array, 
            #                 "right": right_bi_array, 
            #                 "excl_left": excl_left_bi_array,
            #                 "excl_right": excl_right_bi_array}

            incl_left_coverage, incl_right_coverage = np.count_nonzero(left_bi_array), np.count_nonzero(
                right_bi_array)
            excl_left_coverage, excl_right_coverage = np.count_nonzero(excl_left_bi_array), np.count_nonzero(
                excl_right_bi_array)

            # Calculate the MDL gain
            info_theo_scores = self.calculate_mdl_gain(bi_array=right_bi_array, excl_bi_array=excl_right_bi_array,
                                                    icol=icol, cut_option=constant.RIGHT_CUT, log=log)

            # Store info in a dictionary
            right_grow_info = store_grow_info(
                excl_bi_array=excl_right_bi_array, incl_bi_array=right_bi_array, icol=icol,
                cut=cut, cut_option=constant.RIGHT_CUT, incl_mdl_gain=info_theo_scores["absolute_gain"],
                excl_mdl_gain=info_theo_scores["absolute_gain_excl"],
                coverage_excl=excl_right_coverage, coverage_incl=incl_right_coverage,
                normalized_gain_excl=info_theo_scores["absolute_gain_excl"] / excl_right_coverage,
                normalized_gain_incl=info_theo_scores["absolute_gain"] / excl_right_coverage # Question: Shouldn't this be incl_coverage? 
            )

            # Ratio of coverage after the grow step to the coverage of the rule before the grow step
            # This is the gain of the grow step
            if incl_or_excl == "incl":
                right_grow_info[f"coverage_percentage"] = right_grow_info[f"coverage_incl"] / self.coverage
            else:
                right_grow_info[f"coverage_percentage"] = right_grow_info[f"coverage_excl"] / self.coverage_excl                

            
            # Calculate the MDL gain
            info_theo_scores = self.calculate_mdl_gain(bi_array=left_bi_array, excl_bi_array=excl_left_bi_array,
                                                    icol=icol, cut_option=constant.LEFT_CUT, log=log)

            # Store info in a dictionary
            left_grow_info = store_grow_info(
                excl_bi_array=excl_left_bi_array, incl_bi_array=left_bi_array, icol=icol,
                cut=cut, cut_option=constant.LEFT_CUT, incl_mdl_gain=info_theo_scores["absolute_gain"],
                excl_mdl_gain=info_theo_scores["absolute_gain_excl"],
                coverage_excl=excl_left_coverage, coverage_incl=incl_left_coverage,
                normalized_gain_excl=info_theo_scores["absolute_gain_excl"] / excl_left_coverage,
                normalized_gain_incl=info_theo_scores["absolute_gain"] / excl_left_coverage # Question: Shouldn't this be incl_coverage?    
            )

            # Ratio of coverage after the grow step to the coverage of the rule before the grow step
            # This is the gain of the grow step
            if incl_or_excl == "incl":
                left_grow_info[f"coverage_percentage"] = left_grow_info[f"coverage_incl"] / self.coverage
            else:
                left_grow_info[f"coverage_percentage"] = left_grow_info[f"coverage_excl"] / self.coverage_excl

            # Update the beams if the grow step is valid
            # if _validity[f"res_{incl_or_excl}"]:
                # Make sure we do not overwrite a result that has already been written
            assert result_list[result_index] is None
            result_list[result_index] = (left_grow_info, right_grow_info)

            # print(len([1 for result in results if result is not None]))
            if data_info.alg_config.log_learning_process > 2 and log:
                data_info.time_logger.info(f"{data_info.current_rule},{data_info.current_iteration},{data_info.current_candidate},{time.time() - s}")
            
            return (left_grow_info, right_grow_info)
        except Exception as e:
            # log the entire traceback
            # data_info.logger.exception(f"Error in grow_one_literal: {sys.exc_info()}")
            data_info.logger.error(f"Error in grow_one_literal: {e}", exc_info=True)
            raise
        return None
            
    def calculate_mdl_gain(self, bi_array, excl_bi_array, icol, cut_option, log=False):
        """ Calculate the MDL gain when adding a cut to the rule by calling various functions in the model and data encoding.
        
        Parameters
        ---
        bi_array : Array
            Binary array containing the instances covered by the rule, including the instances covered by previous rules
        excl_bi_array : Array
            Binary array containing the instances covered by the rule, excluding the instances covered by previous rules
        icol : int
            Index of the feature for which the cut is being considered
        cut_option : int
            Which type of cut is being considered (LEFT, RIGHT, WITHIN)
            
        Returns
        ---
          : dict
            Containing codelengths and gains for this split
        """
        try:
            cl_model = self.cl_model_after_growing_rule(icol=icol, cut_option=cut_option, log=log)
            cl_data = self.get_cl_data_incl(excl_bi_array=excl_bi_array, incl_bi_array=bi_array)
            cl_data_excl = self.get_cl_data_excl(excl_bi_array)

            absolute_gain = ruleset_info.total_cl - cl_data - cl_model
            absolute_gain_excl = ruleset_info.total_cl - cl_data_excl - cl_model

            return {"cl_model": cl_model, "cl_data": cl_data, "cl_data_excl": cl_data_excl,
                    "absolute_gain": absolute_gain, "absolute_gain_excl": absolute_gain_excl}
        except:
            data_info.logger.error(f"Error in calculate_mdl_gain:")
            raise

    def rule_cl_model_dep(self, condition_matrix, col_orders, log=False):
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
        try:
            # Count the conditions on each feature. Can be 0, 1 or 2. 
            condition_count = (~np.isnan(condition_matrix[0])).astype(int) + (~np.isnan(condition_matrix[1])).astype(int)

            num_variables = np.count_nonzero(condition_count)
            l_num_variables = data_info.cached_cl_model["l_number_of_variables"][num_variables]
            l_which_variables = data_info.cached_cl_model["l_which_variables"][num_variables]

            bool_ = np.ones(data_info.features.shape[0], dtype=bool)
            covered_features = data_info.features[bool_, :]

            l_cuts = 0

            for index, col in enumerate(col_orders):
                s = time.time()
                feature = covered_features[:, [col]]
                up_bound, low_bound = np.max(feature), np.min(feature)

                s = time.time()
                num_cuts = np.count_nonzero((data_info.candidate_cuts[col] >= low_bound) &
                                            (data_info.candidate_cuts[col] <= up_bound))  # only an approximation here.

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

                s = time.time()
                # Now that we have "transmitted" a literal, we can drop data that the rule no longer covers
                # This make the code length of the next literal smaller, as some cutting points might not be relevant anymore
                if index != len(col_orders) - 1:
                    assert condition_count[col] == 1 or condition_count[col] == 2
                    if condition_count[col] == 1:
                        if not np.isnan(condition_matrix[0, col]):
                            bool_ = bool_ & (data_info.features[:, [col]].todense().flatten() <= condition_matrix[0, col])
                        else:
                            bool_ = bool_ & (data_info.features[:, [col]].todense().flatten() > condition_matrix[1, col])
                    else:
                        temp = data_info.features[:, [col]].todense().flatten()
                        bool_ = bool_ & ((temp <= condition_matrix[0, col]) &
                                        (temp > condition_matrix[1, col]))
            return l_num_variables + l_which_variables + l_cuts
        except:
            data_info.logger.error("Error in rule_cl_model_dep")
            raise

    def cl_model_after_growing_rule(self, icol, cut_option, log=False):
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
        try:
            condition_count = np.array(self.condition_bool)
            icols_in_order = self.icols_in_order
            condition_matrix = np.array(self.condition_matrix)

            growing_rule = 0

            # when the rule is still being grown by adding condition using icol, we need to update the condition_count;
            if icol is not None and cut_option is not None:
                if np.isnan(self.condition_matrix[cut_option, icol]):
                    condition_count[icol] += 1
                    # Note that this is just a place holder, to make this position not equal to np.nan; Need to make this more readable later.
                    condition_matrix[0, icol] = np.inf

                if icol not in icols_in_order:
                    icols_in_order = icols_in_order + [icol]
                
                growing_rule = 1

            # note that here is a choice based on the assumption that we can use $X$ to encode the model;
            cl_model_rule_after_growing = self.rule_cl_model_dep(condition_matrix, icols_in_order, log=log)

            # If we are growing a rule, this rule is not in the ruleset yet so we add 1 
            l_num_rules = utils_modelencoding.universal_code_integers(ruleset_info.n_rules + growing_rule)
            
            # Cover redunancy in the order of rules
            cl_redundancy_rule_orders = math.lgamma(ruleset_info.n_rules + 1 + growing_rule) / np.log(2)

            return l_num_rules + cl_model_rule_after_growing - cl_redundancy_rule_orders + ruleset_info.allrules_cl_model
        except:
            data_info.logger.error("Error in cl_model_after_growing_rule")
            raise

    def get_cl_data_excl(self, bool):
        """ Get the code length of the data if rule is added to ruleset.
        Recalculates the else rule code length and adds this rule to allrules_cl_data
        
        Parameters
        ---
        ruleset : Ruleset object
            Ruleset under consideration
        rule : Rule object
            Rule under consideration
        bool : Array
            Boolean array indicating which instances are covered by the rule
        Returns
        ---
        : float
            Code length of the data when this rule is added to the ruleset
        """
        try:
            p_rule = self.calc_probs(self.target_excl[bool])
            coverage_rule = np.count_nonzero(bool)
            negloglike_rule = -coverage_rule * np.sum(np.log2(p_rule[p_rule != 0]) * p_rule[p_rule != 0])

            else_bool = np.array(ruleset_info.uncovered_bool)
            else_bool[self.indices_excl[bool]] = False
            coverage_else = np.count_nonzero(else_bool)
            p_else = utils_calculating_cl.calc_probs(data_info.target[else_bool], data_info.num_class)
            negloglike_else = -coverage_else * np.sum(np.log2(p_else[p_else != 0]) * p_else[p_else != 0])

            regret_else, regret_rule = nml_regret.regret(coverage_else, self.num_class), nml_regret.regret(coverage_rule, self.num_class)

            cl_data = negloglike_else + regret_else + negloglike_rule + regret_rule + ruleset_info.allrules_cl_data

            return cl_data
        except:
            data_info.logger.error("Error in rule_cl_model_dep")
            raise

    def get_cl_data_incl(self, excl_bi_array, incl_bi_array):
        """ Get the code length of the data if rule is added to ruleset.
        Recalculates the else rule code length and adds this rule to allrules_cl_data
        
        Parameters
        ---
        ruleset : Ruleset object
            Ruleset under consideration
        rule : Rule object
            Rule under consideration
        excl_bi_array : Array
            Boolean array indicating which instances are covered by the rule exclusively
        incl_bi_array : Array
            Boolean array indicating which instances are covered by the rule, not necessarily exclusively
        
        Returns
        ---
        : float
            Code length of the data when this rule is added to the ruleset
        """
        try:
            excl_coverage, incl_coverage = np.count_nonzero(excl_bi_array), np.count_nonzero(incl_bi_array)

            p_excl = self.calc_probs(self.target_excl[excl_bi_array])
            p_incl = self.calc_probs(self.target[incl_bi_array])

            both_negloglike = np.zeros(len(modelling_groups),
                                    dtype=float)  # "both" in the name is to emphasize that this is the overlap of both the rule and a modelling_group
            for i, modelling_group in enumerate(modelling_groups):
                # Note: both_negloglike[i] represents negloglike(modelling_group \setdiff rule) + negloglike(modelling_Group \and rule) # noqa
                both_negloglike[i] = modelling_group.evaluate_rule_with_no_updating(indices=self.indices[incl_bi_array])

            # the non-overlapping part for the rule
            non_overlapping_negloglike = -excl_coverage * np.sum(p_excl[p_incl != 0] * np.log2(p_incl[p_incl != 0]))
            rule_regret = nml_regret.regret(incl_coverage, self.num_class)

            new_else_bool = np.zeros(data_info.nrow, dtype=bool)
            new_else_bool[ruleset_info.uncovered_indices] = True
            new_else_bool[self.indices_excl[excl_bi_array]] = False
            new_else_coverage = np.count_nonzero(new_else_bool)
            new_else_p = self.calc_probs(data_info.target[new_else_bool])

            new_else_negloglike = -new_else_coverage * np.sum(np.log2(new_else_p[new_else_p != 0]) * new_else_p[new_else_p != 0])
            new_else_regret = nml_regret.regret(new_else_coverage, data_info.num_class)

            cl_data = (new_else_negloglike + new_else_regret) + (np.sum(both_negloglike) + ruleset_info.allrules_regret) + \
                    (non_overlapping_negloglike + rule_regret)

            return cl_data
        except:
            data_info.logger.error("Error in rule_cl_model_dep")
            raise
    