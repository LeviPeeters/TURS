import numpy as np

def make_rule_from_grow_info(grow_info):
    pass

def extract_rules_from_beams(beams):
    pass

class Ruleset:
    def __init__(self,
                 data_info,
                 data_encoding,
                 model_encoding,
                 constraints = None):
        pass

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
        """ Fit the dat by iteratively adding rules to the ruleset
        
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