import numpy  as np
from functools import partial

import utils_calculating_cl
import DataInfo
import nml_regret
import Rule

class DataEncoding:
    def __init__(self, 
                 data_info: DataInfo.DataInfo
                 ):
        self.data_info = data_info
        self.num_class = data_info.num_class
        self.calc_probs = partial(utils_calculating_cl.calc_probs, num_class=self.num_class)

    @staticmethod
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
    
class NMLencoding(DataEncoding):
    def __init__(self, data_info):
        super().__init__(data_info)
        self.allrules_regret = 0

    def update_ruleset_and_get_cl_data_ruleset_after_adding_rule(self, 
                                                                 ruleset, 
                                                                 rule: Rule.Rule):
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
        ruleset.update_else_rule(rule)  # update Ruleset object's attributes w.r.t the information of the else-rule
        ruleset.elserule_total_cl = self.get_cl_data_elserule(ruleset)  # else-rule's cl_data: negloglike + regret

        allrules_negloglike_except_elserule = ruleset.get_negloglike_all_modelling_groups(rule)
        allrules_regret = np.sum([nml_regret.regret(r.coverage, ruleset.data_info.num_class) for r in ruleset.rules])

        cl_data = ruleset.elserule_total_cl + allrules_negloglike_except_elserule + allrules_regret
        allrules_cl_data = allrules_negloglike_except_elserule + allrules_regret

        self.allrules_regret = allrules_regret

        return [cl_data, allrules_cl_data]

    def get_cl_data_elserule(self, ruleset):
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
        p = self.calc_probs(self.data_info.target[ruleset.uncovered_indices])

        # Because of this below line of code, this function needs to be called after updating the ruleset's attributes
        coverage = len(ruleset.uncovered_indices)

        negloglike_rule = -coverage * np.sum(np.log2(p[p != 0]) * p[p != 0])
        reg = nml_regret.regret(coverage, self.data_info.num_class)
        return negloglike_rule + reg

    def get_cl_data_excl(self, ruleset, rule, bool):
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
            p_rule = self.calc_probs(rule.target_excl[bool])
            coverage_rule = np.count_nonzero(bool)
            negloglike_rule = -coverage_rule * np.sum(np.log2(p_rule[p_rule != 0]) * p_rule[p_rule != 0])

            else_bool = np.array(ruleset.uncovered_bool)
            else_bool[rule.indices_excl[bool]] = False
            coverage_else = np.count_nonzero(else_bool)
            p_else = utils_calculating_cl.calc_probs(self.data_info.target[else_bool], self.data_info.num_class)
            negloglike_else = -coverage_else * np.sum(np.log2(p_else[p_else != 0]) * p_else[p_else != 0])

            regret_else, regret_rule = nml_regret.regret(coverage_else, self.num_class), nml_regret.regret(coverage_rule, self.num_class)

            cl_data = negloglike_else + regret_else + negloglike_rule + regret_rule + ruleset.allrules_cl_data

            return cl_data
        except:
            self.data_info.logger.error("Error in rule_cl_model_dep")
            raise

    def get_cl_data_incl(self, ruleset, rule, excl_bi_array, incl_bi_array):
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

            p_excl = self.calc_probs(rule.target_excl[excl_bi_array])
            p_incl = self.calc_probs(rule.target[incl_bi_array])

            modelling_groups = ruleset.modelling_groups
            both_negloglike = np.zeros(len(modelling_groups),
                                    dtype=float)  # "both" in the name is to emphasize that this is the overlap of both the rule and a modelling_group
            for i, modelling_group in enumerate(modelling_groups):
                # Note: both_negloglike[i] represents negloglike(modelling_group \setdiff rule) + negloglike(modelling_Group \and rule) # noqa
                both_negloglike[i] = modelling_group.evaluate_rule_with_no_updating(indices=rule.indices[incl_bi_array])

            # the non-overlapping part for the rule
            non_overlapping_negloglike = -excl_coverage * np.sum(p_excl[p_incl != 0] * np.log2(p_incl[p_incl != 0]))
            rule_regret = nml_regret.regret(incl_coverage, self.num_class)

            new_else_bool = np.zeros(self.data_info.nrow, dtype=bool)
            new_else_bool[ruleset.uncovered_indices] = True
            new_else_bool[rule.indices_excl[excl_bi_array]] = False
            new_else_coverage = np.count_nonzero(new_else_bool)
            new_else_p = self.calc_probs(self.data_info.target[new_else_bool])

            new_else_negloglike = self.calc_negloglike(p=new_else_p, n=new_else_coverage)
            new_else_regret = nml_regret.regret(new_else_coverage, self.data_info.num_class)

            cl_data = (new_else_negloglike + new_else_regret) + (np.sum(both_negloglike) + self.allrules_regret) + \
                    (non_overlapping_negloglike + rule_regret)

            return cl_data
        except:
            self.data_info.logger.error("Error in rule_cl_model_dep")
            raise