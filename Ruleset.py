import numpy as np
import time
from datetime import datetime
import multiprocessing as mp
import sys
import math
from functools import partial
import dill

import Beam
import ModellingGroup
import DataInfo
import utils_calculating_cl
import utils_modelencoding
import utils
import nml_regret
import constant
from utils_namedtuple import RulesetInfo


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


    rule = Rule(indices=indices, indices_excl=indices_excl,
                rule_base=rule, condition_matrix=condition_matrix, 
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
            if info["coverage_incl"] in coverage_list:
                index_equal = coverage_list.index(info["coverage_incl"])
                if np.all(rules[index_equal].indices == info["_rule"].indices[info["incl_bi_array"]]):
                    continue

            r = make_rule_from_grow_info(grow_info=info)
            rules.append(r)
            coverage_list.append(r.coverage)
    return rules

def setup_worker(data_info_orig, ruleset_info_orig, modelling_groups_orig):
    global ruleset_info
    global data_info
    global modelling_groups

    ruleset_info = ruleset_info_orig
    data_info = data_info_orig
    modelling_groups = modelling_groups_orig

class Ruleset:
    def __init__(self,
                 data_info_orig: DataInfo.DataInfo,
                 constraints = None):

        global data_info
        data_info = data_info_orig

        self.data_info = data_info

        self.rules = []

        self.num_class = data_info.num_class
        self.calc_probs = partial(utils_calculating_cl.calc_probs, num_class=self.num_class)

        self.uncovered_indices = np.arange(data_info.nrow)
        self.uncovered_bool = np.ones(self.data_info.nrow, dtype=bool)
        self.else_rule_p = utils_calculating_cl.calc_probs(target=data_info.target, num_class=data_info.num_class)
        self.else_rule_coverage = self.data_info.nrow
        self.elserule_total_cl = self.get_cl_data_elserule()
        self.majority_class_prior_p = max(utils_calculating_cl.calc_probs(data_info.target, data_info.num_class))

        self.negloglike = -np.sum(data_info.nrow * np.log2(self.else_rule_p[self.else_rule_p != 0]) * self.else_rule_p[self.else_rule_p != 0])
        self.else_rule_negloglike = self.negloglike

        self.cl_data = self.elserule_total_cl
        self.cl_model = 0  # cl model for the whole rule set (including the number of rules)
        self.allrules_cl_model = 0  # cl model for all rules, summed up
        self.total_cl = self.cl_model + self.cl_data  # total cl with 0 rules
        self.allrules_regret = 0

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
            self.update_ruleset_and_get_cl_data_ruleset_after_adding_rule(rule=rule)
        self.cl_model = \
            rule.cl_model_after_growing_rule(icol=None, cut_option=None)

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
        if self.data_info.log_learning_process > 0:
            self.data_info.logger.info(f"ruleset.fit")
            self.data_info.start_time = time.time()

        # Keep track of CL progression
        total_cl = [self.total_cl]

        for iter in range(max_iter):
            self.data_info.current_rule = iter
            if self.data_info.log_learning_process > 0:
                self.data_info.logger.info("--------------------")
                self.data_info.logger.info(f"Iteration {iter}")
                self.data_info.logger.info("--------------------")
            
            # Grow a rule
            rule_to_add = self.search_next_rule(k_consecutively=5)

            # 
            if rule_to_add.incl_gain_per_excl_coverage > 0:
                self.add_rule(rule_to_add)
                total_cl.append(self.total_cl)
                if self.data_info.log_learning_process:
                    self.data_info.logger.info("\n")
                    self.data_info.logger.info(f"Added the following rule to the ruleset:")
                    self.data_info.logger.info(str(rule_to_add))
                    self.data_info.logger.info("\n")
            else:
                break

        if self.data_info.log_learning_process > 0:
            self.data_info.logger.info(f"Finished learning process at {datetime.now().strftime('%Y-%m-%d_%H-%M')}")
            self.data_info.logger.info(f"Total runtime: {time.time() - self.data_info.start_time}")
            self.data_info.logger.info(f"Final ruleset is: ")
            self.data_info.logger.info(str(self))
            self.data_info.logger.info("\n")
            if self.data_info.log_learning_process > 2:
                time_report = utils.time_report(f"./logs/{self.data_info.alg_config.log_folder_name}").to_string()
                self.data_info.logger.info(f"Time report: \n{time_report}")
                utils.time_report_boxplot(f"./logs/{self.data_info.alg_config.log_folder_name}")

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

    def combine_beams(self, beam_list, incl_or_excl):
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

        infos = []
        coverages = []

        # First, we combine the info and coverage lists 
        for beam in beam_list:
            infos.extend([info for info in beam.infos.values() if info is not None])
            coverages.extend([info[f"coverage_{incl_or_excl}"] for info in beam.infos.values() if info is not None])
        
        # Argsort the coverage lists
        argsorted_coverages = np.argsort(coverages)
        groups_coverages = np.array_split([infos[i] for i in argsorted_coverages], self.data_info.beam_width)

        # Now we find the best info for each group
        final_info = []
        for group in groups_coverages:
            if len(group) == 0:
                continue

            # # If probability threshold is set, we remove rules with max probability below the majority class prior
            # # This is used to test whether TURS still performs if we disallow rules that reduce uncertainty in the else rule
            # if self.data_info.alg_config.probability_threshold is True:
                
            #     while len(group) > 0:
            #         best_info_index = np.argmax([info[f"normalized_gain_{incl_or_excl}"] for info in group])
            #         best_info = group[best_info_index]
            #         if max(best_info["_rule"].prob) < self.majority_class_prior_p:
            #         # if max(best_info["_rule"].prob) < 0.97:
            #             # print(f" - {best_info['_rule']} - ")
            #             group = np.delete(group, best_info_index)
            #         else:    
            #             final_info.append(best_info)
            #             break

            # else:
            best_info = group[np.argmax([info[f"normalized_gain_{incl_or_excl}"] for info in group])]
            final_info.append(best_info)

        return final_info

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
        global data_info
        global ruleset_info
        global modelling_groups

        if rule_given is None:
            rule = Rule(indices=np.arange(self.data_info.nrow), 
                        indices_excl=self.uncovered_indices,
                        rule_base=None,
                        condition_matrix=np.repeat(np.nan, self.data_info.ncol * 2).reshape(2, self.data_info.ncol),
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
            self.data_info.current_iteration = i
            if self.data_info.log_learning_process > 0:
                self.data_info.logger.info("")
                self.data_info.logger.info(f"Grow iteration {i}")
                self.data_info.logger.info(f"Number of rules for this iteration: {len(rules_for_next_iter)}")
                self.data_info.logger.info(f"Total number of candidates: {len(rules_candidates)}")
                self.data_info.logger.info("")
            
            # Create a list of candidates for the next iteration
            candidates = []
            for rule in rules_for_next_iter:
                # with dill.detect.trace("dill_trace.log", mode='w'):
                #     dill.dumps(rule)
                # breakpoint()
                for incl_or_excl in ["incl", "excl"]:
                    candidates.append((rule, incl_or_excl))  
                                
            # This is the information that will be passed to the workers
            ruleset_info = RulesetInfo( total_cl=self.total_cl, 
                                    n_rules=len(self.rules), 
                                    allrules_cl_model=self.allrules_cl_model,
                                    uncovered_bool=self.uncovered_bool,
                                    allrules_cl_data=self.allrules_cl_data,
                                    uncovered_indices=self.uncovered_indices,
                                    allrules_regret=self.allrules_regret)
            results = mp.Manager().list()

            if self.data_info.alg_config.workers == -1:
                # No multiprocessing 
                pool = mp.Pool(mp.cpu_count(), initializer=setup_worker, initargs=(self.data_info, ruleset_info, self.modelling_groups))
                setup_worker(self.data_info, ruleset_info, self.modelling_groups)
                for candidate in candidates:
                    candidate[0].grow(results, candidate[1])
            else:
                # Multiprocessing using the specified number of workers
                # pool = mp.Pool(self.data_info.alg_config.workers,  initializer=setup_worker, initargs=(self.data_info, ruleset_info, self.modelling_groups))
            
                s = time.time()
                # res = pool.starmap_async(expand_rule, [(cand[0], cand[1], results) for i, cand in enumerate(candidates)])
                processes = []
                setup_worker(self.data_info, ruleset_info, self.modelling_groups)
                for candidate in candidates:
                    process = mp.Process(target=candidate[0].grow, args=(results, candidate[1]))
                    processes.append(process)
                    process.start()
                
                for process in processes:
                    process.join()

                if self.data_info.log_learning_process > 2:
                    self.data_info.time_logger.info(f"0,{time.time() - s},expand all rules")

            

            beam_list_incl = []
            beam_list_excl = []
            for result in results:
                beam, incl_or_excl = result
                if incl_or_excl == "incl":
                    beam_list_incl.append(beam)
                else:
                    beam_list_excl.append(beam)

            
            # Combine the beams and make GrowInfoBeam objects to store them
            final_info_incl = self.combine_beams(beam_list_incl, "incl")
            final_info_excl = self.combine_beams(beam_list_excl, "excl")
            final_beam_incl = Beam.GrowInfoBeam(width=self.data_info.beam_width)
            final_beam_excl = Beam.GrowInfoBeam(width=self.data_info.beam_width)

            for info in final_info_incl:
                final_beam_incl.update(info, info[f"normalized_gain_incl"])       
            for info in final_info_excl:
                final_beam_excl.update(info, info[f"normalized_gain_excl"])

            


            # If the beams are empty, stop the search
            if len(final_beam_incl.gains) == 0 and len(final_beam_excl.gains) == 0:
                break

            # If the stop condition is met, increment the counter
            stop_condition_element = self.calculate_stop_condition_element(final_beam_incl, final_beam_excl, previous_best_gain, previous_best_excl_gain)
            
            if stop_condition_element:
                counter_worse_best_gain = counter_worse_best_gain + 1
            else:
                counter_worse_best_gain = 0

            # If we found a rule with better gain, store it
            if len(final_beam_incl.gains) > 0:
                previous_best_gain = np.max(final_beam_incl.gains)
            if len(final_beam_excl.gains) > 0:
                previous_best_excl_gain = np.max(final_beam_excl.gains)
            
            # Stop the search if we find no improvement in k_consecutively iterations
            if counter_worse_best_gain > k_consecutively:
                break
            else:
                rules_for_next_iter = extract_rules_from_beams([final_beam_excl, final_beam_incl])
                rules_candidates.extend(rules_for_next_iter)

        # If a probability threshold is set, we remove rules with max probability below the majority class prior
        if self.data_info.alg_config.probability_threshold is True:
            while True:
                which_best_ = np.argmax([r.incl_gain_per_excl_coverage for r in rules_candidates])
                if max(rules_candidates[which_best_].prob) < self.majority_class_prior_p:
                    rules_candidates.pop(which_best_)
                else:
                    break
        else:
            which_best_ = np.argmax([r.incl_gain_per_excl_coverage for r in rules_candidates])
        return rules_candidates[which_best_]


    def update_ruleset_and_get_cl_data_ruleset_after_adding_rule(self, 
                                                                 rule):
        """ Update the ruleset and get the code length of the data and the ruleset after adding a rule
        Question: Why is the else rule updated here, and not in the ruleset class?

        Parameters
        ---
        ruleset : Ruleset object
            Ruleset to be updated
        rule : Rule object
            Rule to be added to the ruleset
        Returns
        ---
        : list
            A list containing the code length of the data and the ruleset after adding the rule
        """
        self.update_else_rule(rule)  # update Ruleset object's attributes w.r.t the information of the else-rule
        self.elserule_total_cl = self.get_cl_data_elserule()  # else-rule's cl_data: negloglike + regret

        allrules_negloglike_except_elserule = self.get_negloglike_all_modelling_groups(rule)
        allrules_regret = np.sum([nml_regret.regret(r.coverage, self.data_info.num_class) for r in self.rules])

        cl_data = self.elserule_total_cl + allrules_negloglike_except_elserule + allrules_regret
        allrules_cl_data = allrules_negloglike_except_elserule + allrules_regret

        self.allrules_regret = allrules_regret

        return [cl_data, allrules_cl_data]

    def get_cl_data_elserule(self):
        """ Get the code length of the data covered by the else rule
        
        Parameters
        ---
        ruleset : Ruleset object
            Ruleset under consideration
        Returns
        ---
        : float
            Code length of the data covered by the else rule
        """
        p = self.calc_probs(self.data_info.target[self.uncovered_indices])

        # Because of this below line of code, this function needs to be called after updating the ruleset's attributes
        coverage = len(self.uncovered_indices)

        negloglike_rule = -coverage * np.sum(np.log2(p[p != 0]) * p[p != 0])
        reg = nml_regret.regret(coverage, self.data_info.num_class)
        return negloglike_rule + reg


    def predict_ruleset(self, X_test):
        """ This function computes the local prediction using a ruleset, given a test set.
        
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
            X_test = X_test.todense()

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
            readable += rule.to_string(verbose=True)
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


def store_grow_info(excl_bi_array, incl_bi_array, icol, cut, cut_option, excl_mdl_gain, incl_mdl_gain,
                    coverage_excl, coverage_incl, normalized_gain_excl, normalized_gain_incl, _rule):
    """ Store the information of a grow step in a dictionary."""
    
    excl_coverage = np.count_nonzero(excl_bi_array)
    incl_coverage = np.count_nonzero(incl_bi_array)
    
    # locals() returns a dictionary containing all local variables
    return locals()

def store_grow_info_rulelist(excl_bi_array, icol, cut, cut_option, excl_normalized_gain):
    return {"excl_bi_array": excl_bi_array, "icol": icol, "cut": cut,
            "cut_option": cut_option, "excl_normalized_gain": excl_normalized_gain}

class Rule:
    def __init__(self, 
                 indices, 
                 indices_excl, 
                 rule_base, 
                 condition_matrix, 
                 mdl_gain, 
                 mdl_gain_excl, 
                 icols_in_order):  

        global data_info
        global modelling_groups
        global ruleset_info

        self.rule_base = rule_base  # The previous level of this rule
        self.icols_in_order = icols_in_order  # The order of the columns in the data

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
        # self.features = data_info.features[indices]
        self.target = data_info.target[indices]
        # self.features_excl = data_info.features[indices_excl]
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

        self.num_class = data_info.num_class
        self.calc_probs = partial(utils_calculating_cl.calc_probs, num_class=self.num_class)

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
        if self.rule_base is None:
            candidate_cuts_selector = (candidate_cuts[icol] < np.max(data_info.features[self.indices_excl, [icol]].flatten())) & \
                                      (candidate_cuts[icol] > np.min(data_info.features[self.indices_excl, [icol]].flatten()))
            candidate_cuts_icol = candidate_cuts[icol][candidate_cuts_selector]
        else:
            candidate_cuts_selector = (candidate_cuts[icol] < np.max(data_info.features[self.indices, [icol]])) & \
                                      (candidate_cuts[icol] > np.min(data_info.features[self.indices, [icol]]))
            candidate_cuts_icol = candidate_cuts[icol][candidate_cuts_selector]
        return candidate_cuts_icol

    def update_grow_beam(self, bi_array, excl_bi_array, icol, cut, cut_option, incl_coverage, excl_coverage,
                         grow_info_beam: Beam.GrowInfoBeam, incl_or_excl, _validity, log=False):
        """ Use information of a grow step to update the beam.

        Parameters
        ---
        bi_array : Array
            Binary array containing the instances covered by the rule, including the instances covered by previous rules
        excl_bi_array : Array
            Binary array containing the instances covered by the rule, excluding the instances covered by previous rules
        icol : int
            Index of the feature for which the cut is being considered
        cut : float
            Cut value for the feature
        cut_option : int
            Which type of cut is being considered (LEFT, RIGHT, WITHIN)
        incl_coverage : int
            Number of instances covered by the rule, including the instances covered by previous rules
        excl_coverage : int
            Number of instances covered by the rule, excluding the instances covered by previous rules
        grow_info_beam : GrowInfoBeam object
            Beam of rules with previously covered instances
        grow_info_beam_excl : GrowInfoBeam object
            Beam of rules without previously covered instances
        _validity : dict
            Results of validity checks for the rule
        log : bool
            Whether to log to a csv file

        Returns
        ---
        None
        """

        s = time.time()
        # Calculate the MDL gain
        info_theo_scores = self.calculate_mdl_gain(bi_array=bi_array, excl_bi_array=excl_bi_array,
                                                   icol=icol, cut_option=cut_option, log=log)
        
        if data_info.alg_config.log_learning_process > 2 and log:
            data_info.time_logger.info(f"0,{time.time() - s}, MDL gain ")

        # Store info in a dictionary
        grow_info = store_grow_info(
            excl_bi_array=excl_bi_array, incl_bi_array=bi_array, icol=icol,
            cut=cut, cut_option=cut_option, incl_mdl_gain=info_theo_scores["absolute_gain"],
            excl_mdl_gain=info_theo_scores["absolute_gain_excl"],
            coverage_excl=excl_coverage, coverage_incl=incl_coverage,
            normalized_gain_excl=info_theo_scores["absolute_gain_excl"] / excl_coverage,
            normalized_gain_incl=info_theo_scores["absolute_gain"] / excl_coverage, # Question: Shouldn't this be incl_coverage?    
            _rule=self
        )

        # Ratio of coverage after the grow step to the coverage of the rule before the grow step
        # This is the gain of the grow step
        if incl_or_excl == "incl":
            cov_percent = grow_info[f"coverage_incl"] / self.coverage
        else:
            cov_percent = grow_info[f"coverage_excl"] / self.coverage_excl

        # Update the beams if the grow step is valid
        # Be careful when multithreading here! Two threads should not update the beam at the same time
        if _validity[f"res_{incl_or_excl}"]:
            grow_info_beam.update(grow_info, grow_info[f"normalized_gain_{incl_or_excl}"], cov_percent)
        

    def grow(self, results, incl_or_excl):
        """ Grow the rule by one step and update the 
        
        Parameters
        ---
        grow_info_beam : DiverseCovBeam object
            Beam object to store information for this grow step
        grow_info_beam_excl : DiverseCovBeam object
            Beam of object to store information for this grow step without considering previously covered instances
            
        Returns
        ---
        None
        """
        s2 = time.time()
        if data_info.alg_config.log_learning_process > 0:
            data_info.growth_logger.info(f"{data_info.current_rule},{data_info.current_iteration},{self.coverage},{self.coverage_excl},{self.mdl_gain},{self.mdl_gain_excl}")
        if data_info.alg_config.log_learning_process > 1:    
            data_info.logger.info(str(self))
        
        grow_info_beam = Beam.DiverseCovBeam(width=data_info.beam_width)
        
        candidate_cuts = data_info.candidate_cuts

        total_time_getting_data = 0

        # Consider each feature
        for icol in range(data_info.ncol):
            candidate_cuts_icol = self.get_candidate_cuts_icol_given_rule(candidate_cuts, icol)

            s = time.time()
            # Sparse: The binary array needs to be converted to dense and flattened, as sparse matrices do not reduce in dimension after slicing
            bi_array = data_info.features[:, [icol]].todense().flatten()
            total_time_getting_data += time.time() - s

            # Consider every candidate cut point
            for i, cut in enumerate(candidate_cuts_icol):
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
                _validity = self.validity_check(icol=icol, cut=cut, bi_arrays=bi_arrays)

                if data_info.not_use_excl_:
                    _validity["res_excl"] = False

                if _validity["res_excl"] == False and _validity["res_incl"] == False:
                    continue

                incl_left_coverage, incl_right_coverage = np.count_nonzero(left_bi_array), np.count_nonzero(
                    right_bi_array)
                excl_left_coverage, excl_right_coverage = np.count_nonzero(excl_left_bi_array), np.count_nonzero(
                    excl_right_bi_array)

                # Question: Why is there no check on incl_coverage being 0?
                if excl_left_coverage == 0 or excl_right_coverage == 0:
                    continue
                
                # Update the beam with the results. We do this twice, because a cut can be < or >
                self.update_grow_beam(bi_array=left_bi_array, excl_bi_array=excl_left_bi_array, icol=icol,
                                      cut=cut, cut_option=constant.LEFT_CUT,
                                      incl_coverage=incl_left_coverage, excl_coverage=excl_left_coverage,
                                      grow_info_beam=grow_info_beam, incl_or_excl=incl_or_excl,
                                      _validity=_validity)

                self.update_grow_beam(bi_array=right_bi_array, excl_bi_array=excl_right_bi_array, icol=icol,
                                      cut=cut, cut_option=constant.RIGHT_CUT,
                                      incl_coverage=incl_right_coverage, excl_coverage=excl_right_coverage,
                                      grow_info_beam=grow_info_beam, incl_or_excl=incl_or_excl,
                                      _validity=_validity)

        results.append((grow_info_beam, incl_or_excl))

        if data_info.alg_config.log_learning_process > 2:
            data_info.time_logger.info(f"0,{time.time()-s2},rule grow")
        


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
        cl_model = self.cl_model_after_growing_rule(icol=icol, cut_option=cut_option, log=log)
        cl_data = self.get_cl_data_incl(excl_bi_array=excl_bi_array, incl_bi_array=bi_array)
        cl_data_excl = self.get_cl_data_excl(excl_bi_array)
        
        absolute_gain = ruleset_info.total_cl - cl_data - cl_model
        absolute_gain_excl = ruleset_info.total_cl - cl_data_excl - cl_model

        return {"cl_model": cl_model, "cl_data": cl_data, "cl_data_excl": cl_data_excl,
                "absolute_gain": absolute_gain, "absolute_gain_excl": absolute_gain_excl}

    def calc_negloglike(p, n):
        """ Calculate the negative log-likelihood of a set of instances
        Parameters
        ---
        p : Array
            Vector containing the probability of encountering each target
        n : int
            Number of instances
        Returns
        ---
        negloglike : float
            Negative log-likelihood of the set of instances
        """
        return -n * np.sum(np.log2(p[p != 0]) * p[p != 0])

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
        s = time.time()
        # Count the conditions on each feature. Can be 0, 1 or 2. 
        condition_count = (~np.isnan(condition_matrix[0])).astype(int) + (~np.isnan(condition_matrix[1])).astype(int)

        num_variables = np.count_nonzero(condition_count)
        l_num_variables = data_info.cached_cl_model["l_number_of_variables"][num_variables]
        l_which_variables = data_info.cached_cl_model["l_which_variables"][num_variables]

        bool_ = np.ones(data_info.features.shape[0], dtype=bool)

        covered_features = data_info.features[bool_, :]

        l_cuts = 0

        if data_info.log_learning_process > 2 and log:
            data_info.time_logger.info(f"0,{time.time() - s},CL model -> before loop")

        for index, col in enumerate(col_orders):
            s = time.time()
            feature = covered_features[:, [col]]
            up_bound, low_bound = np.max(feature), np.min(feature)
            if data_info.log_learning_process > 2 and log:
                data_info.time_logger.info(f"0,{time.time() - s},CL model -> calculate bounds")

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

            if data_info.log_learning_process > 2 and log:
                data_info.time_logger.info(f"0,{time.time() - s},CL model -> misc")

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
            if data_info.log_learning_process > 2 and log:
                data_info.time_logger.info(f"0,{time.time() - s}, CL model-> drop uncovered data")

        return l_num_variables + l_which_variables + l_cuts

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

    def validity_check(self, icol, cut, bi_arrays):
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
        res_excl = True
        res_incl = True
        if data_info.alg_config.validity_check == "no_check":
            pass
        elif data_info.alg_config.validity_check == "excl_check":
            res_excl = self.check_split_validity_excl(icol, bi_arrays)
        elif data_info.alg_config.validity_check == "incl_check":
            res_incl = self.check_split_validity(icol, bi_arrays)
        elif data_info.alg_config.validity_check == "either":
            res_excl = self.check_split_validity_excl(icol, bi_arrays)
            res_incl = self.check_split_validity(icol, bi_arrays)
        else:
            # TODO: I should improve this error message a bit
            sys.exit("Error: the if-else statement should not end up here")
        return {"res_excl": res_excl, "res_incl": res_incl}

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
        indices_left, indices_right = np.where(bi_arrays["left"])[0], np.where(bi_arrays["right"])[0]

        # Compute probabilities for the coverage of the rule and both sides of the split
        p_left = utils_calculating_cl.calc_probs(data_info.target[indices_left], data_info.num_class)
        p_right = utils_calculating_cl.calc_probs(data_info.target[indices_right], data_info.num_class)

        # Compute the negative log-likelihoods for these probabilities
        nll_rule = utils_calculating_cl.calc_negloglike(self.prob, self.coverage)
        nll_left = utils_calculating_cl.calc_negloglike(p_left, len(indices_left))
        nll_right = utils_calculating_cl.calc_negloglike(p_right, len(indices_right))

        # The extra model code length this split would add
        cl_model_extra = data_info.cached_cl_model["l_cut"][0][icol]  # 0 represents for the "one split cut", instead of "two-splits cut"
        cl_model_extra += np.log2(data_info.data_ncol_for_encoding)
        num_vars = np.sum(self.condition_bool > 0)
        cl_model_extra += data_info.cached_cl_model["l_number_of_variables"][num_vars + 1] - \
                        data_info.cached_cl_model["l_number_of_variables"][num_vars]

        # if this validity score is positive, the split decreases the total code length
        validity = nll_rule + nml_regret.regret(self.coverage, data_info.num_class) \
                    - nll_left - nml_regret.regret(len(indices_left), data_info.num_class) \
                    - nll_right - nml_regret.regret(len(indices_right), data_info.num_class) \
                    - cl_model_extra

        return (validity > 0)

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
        indices_left, indices_right = np.where(bi_arrays["excl_left"])[0], np.where(bi_arrays["excl_right"])[0]

        # Compute probabilities for the coverage of the rule and both sides of the split
        p_left = utils_calculating_cl.calc_probs(data_info.target[indices_left], data_info.num_class)
        p_right = utils_calculating_cl.calc_probs(data_info.target[indices_right], data_info.num_class)

        # Compute the negative log-likelihoods for these probabilities
        nll_rule = utils_calculating_cl.calc_negloglike(self.prob_excl, self.coverage_excl)
        nll_left = utils_calculating_cl.calc_negloglike(p_left, len(indices_left))
        nll_right = utils_calculating_cl.calc_negloglike(p_right, len(indices_right))

        # The extra model code length this split would add
        cl_model_extra = data_info.cached_cl_model["l_cut"][0][icol]  # 0 represents for the "one split cut", instead of "two-splits cut"
        cl_model_extra += np.log2(data_info.data_ncol_for_encoding)
        num_vars = np.sum(self.condition_bool > 0)
        cl_model_extra += data_info.cached_cl_model["l_number_of_variables"][num_vars + 1] - \
                        data_info.cached_cl_model["l_number_of_variables"][num_vars]

        # if this validity score is positive, the split decreases the total code length
        validity = nll_rule + nml_regret.regret(self.coverage_excl, data_info.num_class) \
                    - nll_left - nml_regret.regret(len(indices_left), data_info.num_class) \
                    - nll_right - nml_regret.regret(len(indices_right), data_info.num_class) \
                    - cl_model_extra

        return (validity > 0)

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
        feature_names = data_info.feature_names
        label_names = data_info.alg_config.label_names
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
            if len(values) < len(data_info.categorical_features[feature]) / 2:
                readable += f"{feature} != {', '.join(values)};    "
            else:
                readable += f"{feature} == {', '.join([v for v in data_info.categorical_features[feature] if v not in values])};    "

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
