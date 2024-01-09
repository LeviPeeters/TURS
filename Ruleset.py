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
    pass

def extract_rules_from_beams(beams):
    pass

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
        
        pass

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
        pass

    def combine_beams(self, incl_beam_list, excl_beam_list):
        """ Combine multiple beams TODO
        
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
        pass

    def search_next_rule(self, k_consecutively, rule_given=None):
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
        pass