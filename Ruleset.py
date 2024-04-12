import numpy as np
import os
from datetime import datetime
import logging

import Rule
import Beam
import ModellingGroup
import DataInfo
import utils_calculating_cl
import ModelEncoding
import DataEncoding

def make_rule_from_grow_info(grow_info):
    """ Make a rule from the information of a grow step

    Parameters
    ---
    grow_info : Dict
        Information about the grow step
    
    Returns
    ---
    rule : Rule object
        Rule created from the grow step information
    """

    rule = grow_info["_rule"]
    indices = rule.indices[grow_info["incl_bi_array"]]
    indices_excl = rule.indices_excl[grow_info["excl_bi_array"]]

    # Update the condition matrix
    condition_matrix = np.array(rule.condition_matrix)
    condition_matrix[grow_info["cut_option"], grow_info["icol"]] = grow_info["cut"]
    
    # Add the column to the list of columns in order if it's not there yet
    if grow_info["icol"] in rule.icols_in_order:
        new_icols_in_order = rule.icols_in_order
    else:
        new_icols_in_order = rule.icols_in_order + [grow_info["icol"]]


    rule = Rule.Rule(indices=indices, indices_excl=indices_excl, data_info=rule.data_info,
                rule_base=rule, condition_matrix=condition_matrix, ruleset=rule.ruleset,
                mdl_gain_excl=grow_info["excl_mdl_gain"],
                mdl_gain=grow_info["incl_mdl_gain"],
                icols_in_order=new_icols_in_order)
    return rule

def extract_rules_from_beams(beams):
    ''' Extract rules from beams

    Parameters
    ---
    beams : list of Beam objects
        List of beams from which the rules are extracted
    
    Returns
    ---
    rules : list of Rule objects
    '''
    rules = []
    coverage_list = []
    for beam in beams:
        for info in beam.infos:
            # Check if there is already a rule with the same coverage
            # TODO there must be a better way to do this
            if info["coverage_incl"] in coverage_list:
                index_equal = coverage_list.index(info["coverage_incl"])
                if np.all(rules[index_equal].indices == info["_rule"].indices[info["incl_bi_array"]]):
                    continue

            r = make_rule_from_grow_info(grow_info=info)
            rules.append(r)
            coverage_list.append(r.coverage)
    return rules


class Ruleset:
    def __init__(self,
                 data_info: DataInfo.DataInfo,
                 data_encoding: DataEncoding.NMLencoding,
                 model_encoding: ModelEncoding.ModelEncodingDependingOnData,
                 constraints = None):

        self.data_info = data_info
        self.model_encoding = model_encoding
        self.data_encoding = data_encoding

        self.rules = []

        self.uncovered_indices = np.arange(data_info.nrow)
        self.uncovered_bool = np.ones(self.data_info.nrow, dtype=bool)
        self.else_rule_p = utils_calculating_cl.calc_probs(target=data_info.target, num_class=data_info.num_class)
        self.else_rule_coverage = self.data_info.nrow
        self.elserule_total_cl = self.data_encoding.get_cl_data_elserule(ruleset=self)

        self.negloglike = -np.sum(data_info.nrow * np.log2(self.else_rule_p[self.else_rule_p != 0]) * self.else_rule_p[self.else_rule_p != 0])
        self.else_rule_negloglike = self.negloglike

        self.cl_data = self.elserule_total_cl
        self.cl_model = 0  # cl model for the whole rule set (including the number of rules)
        self.allrules_cl_model = 0  # cl model for all rules, summed up
        self.total_cl = self.cl_model + self.cl_data  # total cl with 0 rules

        self.modelling_groups = []

        self.allrules_cl_data = 0

        if constraints is None:
            self.constraints = {}
        else:
            self.constraints = constraints

        
        

    def add_rule(self, rule):
        """ Add a rule to the ruleset and update code length accordingly
        
        Parameters
        ---
        rule : Rule object
            Rule to be added to the ruleset
        """

        # Append the rule
        self.rules.append(rule)
        
        self.cl_data, self.allrules_cl_data = \
            self.data_encoding.update_ruleset_and_get_cl_data_ruleset_after_adding_rule(ruleset=self, rule=rule)
        self.cl_model = \
            self.model_encoding.cl_model_after_growing_rule(rule=rule, ruleset=self, icol=None, cut_option=None)

        # Update total codelength
        self.total_cl = self.cl_data + self.cl_model
        self.allrules_cl_model += rule.cl_model

    def update_else_rule(self, rule):
        """ Update properties of the else rule

        Parameters
        ---
        rule : Rule object
            Rule to be added to the ruleset
        
        Returns
        ---
        None
        """
        self.uncovered_bool = np.bitwise_and(self.uncovered_bool, ~rule.bool_array)
        self.uncovered_indices = np.where(self.uncovered_bool)[0]
        self.else_rule_coverage = len(self.uncovered_indices)
        self.else_rule_p = utils_calculating_cl.calc_probs(self.data_info.target[self.uncovered_indices], self.data_info.num_class)
        self.else_rule_negloglike = utils_calculating_cl.calc_negloglike(self.else_rule_p, self.else_rule_coverage)

    def get_negloglike_all_modelling_groups(self, rule):
        """ Calculate the total negative log-likelihood of all rules when adding a new rule
        
        Parameters
        ---
        rule : Rule object
            Rule to be added
            
        Returns
        ---
        negloglike : float
            Summed negative log-likelihood of all modelling groups for the given rule
        """
        
        all_mg_negloglike = []
        if len(self.modelling_groups) == 0:
            mg = ModellingGroup.ModellingGroup(data_info=self.data_info, bool_cover=rule.bool_array,
                                bool_use_for_model=rule.bool_array,
                                rules_involved=[0], prob_model=rule.prob,
                                prob_cover=rule.prob)
            all_mg_negloglike.append(mg.negloglike)
            self.modelling_groups.append(mg)
        else:
            num_mgs = len(self.modelling_groups)
            for jj in range(num_mgs):
                m = self.modelling_groups[jj]
                evaluate_res = m.evaluate_rule(rule, update_rule_index=len(self.rules) - 1)
                all_mg_negloglike.append(evaluate_res[0])
                if evaluate_res[1] is not None:
                    self.modelling_groups.append(evaluate_res[1])

            mg = ModellingGroup.ModellingGroup(data_info=self.data_info,
                                bool_cover=rule.bool_array_excl,
                                bool_use_for_model=rule.bool_array,
                                rules_involved=[len(self.rules) - 1], prob_model=rule.prob,
                                prob_cover=rule.prob_excl)
            all_mg_negloglike.append(mg.negloglike)
            self.modelling_groups.append(mg)
        return np.sum(all_mg_negloglike)

    def fit(self, max_iter=1000):
        """ Fit the data by iteratively adding rules to the ruleset
        # TODO: Think about logging
        
        Parameters
        ---
        max_iter : int
            Maximum number of iterations
        printing : bool
            Whether to print the results of each iteration
            
        Returns 
        ---
        total_cl : float
            Total code length of the ruleset
        """
        logging.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ruleset.fit")

        # Keep track of CL progression
        total_cl = [self.total_cl]

        for iter in range(max_iter):
            if self.data_info.log_learning_process:
                logging.info("--------------------")
                logging.info(f"Iteration {iter}")
                logging.info("--------------------")
            
            # Grow a rule
            rule_to_add = self.search_next_rule(k_consecutively=5)

            # 
            if rule_to_add.incl_gain_per_excl_coverage > 0:
                self.add_rule(rule_to_add)
                total_cl.append(self.total_cl)
                if self.data_info.log_learning_process:
                    logging.info(f"Added the following rule to the ruleset:")
                    logging.info(str(rule_to_add))
            else:
                break

        if self.data_info.log_learning_process:
            logging.info(f"Finished learning process at {datetime.now().strftime('%Y-%m-%d_%H-%M')}")
            logging.info(f"Final ruleset is: ")
            logging.info(str(self))
            logging.info("\n")
    
        return total_cl

        

    @staticmethod
    def calculate_stop_condition_element(incl_beam, excl_beam, prev_best_gain, prev_best_excl_gain):
        """ Calculate the stop condition element for a given beam
        
        Parameters
        ---
        incl_beam : GrowInfoBeam object
            Beam of rules with previously covered instances
        excl_beam : GrowInfoBeam object
            Beam of rules without previously covered instances
        prev_best_gain : float
            Previous best gain with previously covered instances
        prev_best_excl_gain : float
            Previous best gain without previously covered instances 
            
        Returns
        ---
        : bool
            Whether the stop condition is met
        """
        # For incl and excl beam, check if the max gain is lower than the previous best gain
        condition1 = len(incl_beam.gains) > 0 and np.max(incl_beam.gains) < prev_best_gain
        condition2 = len(excl_beam.gains) > 0 and np.max(excl_beam.gains) < prev_best_excl_gain
        
        # Check if any rule has a negative gain. This should never happen, as we only add rules with positive gain
        condition3 = len(excl_beam.gains) > 0 and len(incl_beam.gains) > 0 and np.max(incl_beam.gains) < 0 and np.max(excl_beam.gains) < 0
        return (condition1 and condition2) or condition3

    def combine_beams(self, incl_beam_list, excl_beam_list):
        """ Receives a list of beams, each containing the information from a grow step for a rule that's being grown
        Finds the best rule in each coverage interval over all beams
        
        Parameters
        ---
        incl_beam_list : list
            List of GrowInfoBeam objects with previously covered instances
        excl_beam_list : list
            List of GrowInfoBeam objects without previously covered instances
        
        Returns
        ---
        final_info_incl : list
            list of info dicts for the best rules, including previously covered instances
        final_info_excl : list
            list of info dicts for the best rules, excluding previously covered instances
        """
        logging.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ruleset.combine_beams")

        infos_incl, infos_excl = [], []
        coverages_incl, coverages_excl = [], []

        # First, we combine the info and coverage lists for the incl beam
        for incl_beam in incl_beam_list:
            infos_incl.extend([info for info in incl_beam.infos.values() if info is not None])
            coverages_incl.extend([info["coverage_incl"] for info in incl_beam.infos.values() if info is not None])
        
        # Argsort the coverage lists
        argsorted_coverages_incl = np.argsort(coverages_incl)
        groups_coverages_incl = np.array_split([infos_incl[i] for i in argsorted_coverages_incl], self.data_info.beam_width)

        # Now we find the best info for each group
        final_info_incl = []
        for group in groups_coverages_incl:
            if len(group) == 0:
                continue
            final_info_incl.append(group[np.argmax([info["normalized_gain_incl"] for info in group])])

        # Repeat the process for the excl beam
        for excl_beam in excl_beam_list:
            infos_excl.extend([info for info in excl_beam.infos.values() if info is not None])
            coverages_excl.extend([info["coverage_excl"] for info in excl_beam.infos.values() if info is not None])
        argsorted_coverages_excl = np.argsort(coverages_excl)
        groups_coverages_excl = np.array_split([infos_excl[i] for i in argsorted_coverages_excl], self.data_info.beam_width)
        
        final_info_excl = []
        for group in groups_coverages_excl:
            if len(group) == 0:
                continue
            final_info_excl.append(group[np.argmax([info["normalized_gain_excl"] for info in group])])
        
        return final_info_incl, final_info_excl

    def search_next_rule(self, 
                         k_consecutively, 
                         rule_given = None
                         ):
        """ Execute a beam search to find the next rule to add to the ruleset
        
        Parameters
        ---
        k_consecutively : int
            Number of iterations without improvement before stopping the beam search
        rule_given : Rule object
            Rule to be expanded using beam search. If None, the search starts with an empty rule
        
        Returns
        ---
          : Rule object
            Rule found by the beam search
        """
        logging.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ruleset.search_next_rule")

        if rule_given is None:
            rule = Rule.Rule(indices=np.arange(self.data_info.nrow), 
                        indices_excl=self.uncovered_indices,
                        data_info=self.data_info, 
                        rule_base=None,
                        condition_matrix=np.repeat(np.nan, self.data_info.ncol * 2).reshape(2, self.data_info.ncol),
                        ruleset=self, 
                        mdl_gain=-np.Inf, 
                        mdl_gain_excl=-np.Inf, 
                        icols_in_order=[])
        else:
            rule = rule_given  

        rules_for_next_iter = [rule]
        rules_candidates = [rule]

        previous_best_gain, previous_best_excl_gain = -np.Inf, -np.Inf
        counter_worse_best_gain = 0

        for i in range(self.data_info.max_grow_iter):
            if self.data_info.log_learning_process:
                logging.info(f"    Grow iteration {i}")
                logging.info(f"    Number of rules for this iteration: {len(rules_for_next_iter)}")
                logging.info(f"    Total number of candidates: {len(rules_candidates)}")

            excl_beam_list, incl_beam_list = [], []

            # For each rule, initialize a beam and put information for a grow step in it
            for rule in rules_for_next_iter:
                excl_beam = Beam.DiverseCovBeam(width=self.data_info.beam_width)
                incl_beam = Beam.DiverseCovBeam(width=self.data_info.beam_width)
                rule.grow(grow_info_beam=incl_beam, grow_info_beam_excl=excl_beam)
                excl_beam_list.append(excl_beam)
                incl_beam_list.append(incl_beam)
            
            # Combine the beams and make GrowInfoBeam objects to store them
            final_info_incl, final_info_excl = self.combine_beams(incl_beam_list, excl_beam_list)
            final_incl_beam = Beam.GrowInfoBeam(width=self.data_info.beam_width)
            final_excl_beam = Beam.GrowInfoBeam(width=self.data_info.beam_width)

            for info in final_info_incl:
                final_incl_beam.update(info, info["normalized_gain_incl"])
            for info in final_info_excl:
                final_excl_beam.update(info, info["normalized_gain_excl"])

            # If the beams are empty, stop the search
            if len(final_incl_beam.gains) == 0 and len(final_excl_beam.gains) == 0:
                break

            # If the stop condition is met, increment the counter
            stop_condition_element = self.calculate_stop_condition_element(final_incl_beam, final_excl_beam, previous_best_gain, previous_best_excl_gain)
            
            if stop_condition_element:
                counter_worse_best_gain = counter_worse_best_gain + 1
            else:
                counter_worse_best_gain = 0

            # If we found a rule with better gain, store it
            if len(final_incl_beam.gains) > 0:
                previous_best_gain = np.max(final_incl_beam.gains)
            if len(final_excl_beam.gains) > 0:
                previous_best_excl_gain = np.max(final_excl_beam.gains)

            # Stop the search if we find no improvement in k_consecutively iterations
            if counter_worse_best_gain > k_consecutively:
                break
            else:
                rules_for_next_iter = extract_rules_from_beams([final_excl_beam, final_incl_beam])
                rules_candidates.extend(rules_for_next_iter)

        which_best_ = np.argmax([r.incl_gain_per_excl_coverage for r in rules_candidates])
        return rules_candidates[which_best_]

    def predict_ruleset(self, X_test):
        """ This function computes the local prediction using a ruleset, given a test set.
        TODO: This does not work with sparse matrices yet
        
        Parameters
        ----------
        ruleset : RuleSet
            The ruleset for which we want to compute the local prediction.
        X_test : np.array
            The test set.
        
        Returns
        -------
        : Array
            Probability distributions for the prediction
        """
        if type(X_test) != np.ndarray:
            X_test = X_test.to_numpy()

        prob_predicted = np.zeros((len(X_test), self.data_info.num_class), dtype=float)
        cover_matrix = np.zeros((len(X_test), len(self.rules) + 1), dtype=bool)

        test_uncovered_bool = np.ones(len(X_test), dtype=bool)
        for ir, rule in enumerate(self.rules):
            r_bool_array = np.ones(len(X_test), dtype=bool)

            condition_matrix = np.array(rule.condition_matrix)
            condition_bool = np.array(rule.condition_bool)
            which_vars = np.where(condition_bool > 0)[0]

            upper_bound, lower_bound = condition_matrix[0], condition_matrix[1]
            upper_bound[np.isnan(upper_bound)] = np.Inf
            lower_bound[np.isnan(lower_bound)] = -np.Inf

            for v in which_vars:
                r_bool_array = r_bool_array & (X_test[:, v] < upper_bound[v]) & (X_test[:, v] >= lower_bound[v])

            cover_matrix[:, ir] = r_bool_array
            test_uncovered_bool = test_uncovered_bool & ~r_bool_array
        cover_matrix[:, -1] = test_uncovered_bool

        # From chatGPT: "I use dtype=object to allow the elements of powers_of_2 and binary_vector to be Python integers which can handle arbitrary large values."
        # That is, by using "object", the element of numpy array becomes Python Int, instead of numpy.int64;
        cover_matrix_int = cover_matrix.astype(int)
        unique_id = np.zeros(len(X_test), dtype=object)
        power_of_two = 2 ** np.arange(cover_matrix_int.shape[1], dtype=object)
        for kol in range(cover_matrix_int.shape[1]):
            # unique_id += 2 ** kol * cover_matrix_int[:, kol]    # This may fail when 2 ** kol becomes very large
            unique_id += power_of_two[kol] * cover_matrix_int[:, kol].astype(object)

        groups, ret_index = np.unique(unique_id, return_index=True)
        unique_id_dir = {}
        for g, rind in zip(groups, ret_index):
            unique_id_dir[g] = cover_matrix_int[rind]

        unique_id_prob_dir = {}
        for z, t in unique_id_dir.items():
            bool_model = np.zeros(len(self.data_info.target), dtype=bool)
            for i_tt, tt in enumerate(t):
                if tt == 1:
                    if i_tt == len(self.rules):
                        bool_model = self.uncovered_bool
                    else:
                        bool_model = np.bitwise_or(bool_model, self.rules[i_tt].bool_array)
            unique_id_prob_dir[z] = utils_calculating_cl.calc_probs(self.data_info.target[bool_model],
                                            self.data_info.num_class)

        for i in range(len(prob_predicted)):
            prob_predicted[i] = unique_id_prob_dir[unique_id[i]]

        return prob_predicted

    def __str__(self):
        """ This function prints a ruleset in a readable way.
    
        Parameters
        ---
        ruleset : RuleSet
            The ruleset to be printed.
        
        Returns
        ---
        None
        """
        readable = ""
        label_names = self.data_info.alg_config.label_names
        for rule in self.rules:
            readable += str(rule)
            readable += "\n"
        readable += "If none of above,\n"
        readable += "Then:\n"
        if len(self.else_rule_p) > 5:
            readable += f"Highest probability is {max(self.else_rule_p)} for outcome {label_names[np.argmax(self.else_rule_p)]}"
        else:
            for i in range(len(label_names)):
                readable += f"Probability of {label_names[i]} is {round(self.else_rule_p[i], 2)}\n"
        readable += f"Coverage of the else rule: {self.else_rule_coverage}\n"

        return readable
