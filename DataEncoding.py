import numpy  as np
from functools import partial

import utils_calculating_cl
import DataInfo
import nml_regret

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

    def update_ruleset_and_get_cl_data_ruleset_after_adding_rule(self, ruleset, rule):
        """ Update the ruleset and get the code length of the data and the ruleset after adding a rule

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
        pass

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
        """ Get the code length of the data if rule is added to ruleset
        
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
        pass

    def get_cl_data_incl(self, ruleset, rule, excl_bi_array, incl_bi_array):
        """ TODO: This isn't quite clear to me
        Get the code length of the data if rule is added to ruleset
        
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
        pass
