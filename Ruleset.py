import numpy as np

import Rule
import Beam
import ModellingGroup
import DataInfo
import utils_readable
import utils_predict
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
    indices_excl_overlap = rule.indices_excl_overlap[grow_info["excl_bi_array"]]

    condition_matrix = np.array(rule.condition_matrix)
    condition_matrix[grow_info["cut_option"], grow_info["icol"]] = grow_info["cut"]
    if grow_info["icol"] in rule.icols_in_order:
        new_icols_in_order = rule.icols_in_order
    else:
        new_icols_in_order = rule.icols_in_order + [grow_info["icol"]]
    rule = Rule(indices=indices, indices_excl_overlap=indices_excl_overlap, data_info=rule.data_info,
                rule_base=rule, condition_matrix=condition_matrix, ruleset=rule.ruleset,
                excl_mdl_gain=grow_info["excl_mdl_gain"],
                incl_mdl_gain=grow_info["incl_mdl_gain"],
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
        
        self.log_folder_name = None

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
        pass

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
        pass

    def get_negloglike_all_modelling_groups(self, rule):
        """ Calculate the negative log-likelihood of the modelling groups for a given rule
        
        Parameters
        ---
        rule : Rule object
            Rule for which the negative log-likelihood of the modelling groups is calculated
            
        Returns
        ---
        negloglike : float
            Summed negative log-likelihood of all modelling groups for the given rule
        """
        pass

    def fit(self, max_iter=1000, printing=True):
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

        # Keep track of CL progression
        total_cl = [self.total_cl]

        for iter in range(max_iter):
            if printing:
                print("Iteration: ", iter)
            
            # Grow a rule
            rule_to_add = self.search_next_rule(k_consecutively=5)
            

        

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
        Finds the 
        
        Parameters
        ---
        incl_beam_list : list
            List of GrowInfoBeam objects with previously covered instances
        excl_beam_list : list
            List of GrowInfoBeam objects without previously covered instances
        
        Returns
        ---
        final_info_incl : TODO
        final_info_excl : TODO
        """
        infos_incl, infos_excl = [], []
        coverages_incl, coverages_excl = [], []

        # First, we combine the info and coverage lists for the incl beam
        for incl_beam in incl_beam_list:
            infos_incl.extend([info for info in incl_beam.infos.values() if info is not None])
            coverages_incl.extend([info["coverage_incl"] for info in incl_beam.infos.values() if info is not None])
        
        # Argsort the coverage lists
        # QUESTION: Why is this array_split here
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
                         rule_given: Rule.Rule = None
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
        if rule_given is None:
            rule = Rule(indices=np.arange(self.data_info.nrow), 
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

        # TODO: Figure out what max_grow_iter is
        for i in range(self.data_info.max_grow_iter):
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
