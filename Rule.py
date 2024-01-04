import numpy as np

class Rule:
    def __init__(self, 
                 indices, 
                 indices_excl_overlap, 
                 data_info, 
                 rule_base, 
                 condition_matrix, 
                 ruleset,
                 excl_mdl_gain, 
                 incl_mdl_gain, 
                 icols_in_order):
        pass

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
        pass

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
        pass

    def get_candidate_cuts_icol_given_rule(self, candidate_cuts, icol):
        """ Given a list of candidate cuts and an index of a feature, return the candidate cuts for that feature.
        
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
        pass

    def update_grow_beam(self, bi_array, excl_bi_array, icol, cut, cut_option, incl_coverage, excl_coverage,
                         grow_info_beam, grow_info_beam_excl, _validity):
        """ Use rule growing information to update the beam.

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

        Returns
        ---
        None
        """

    def grow(self, grow_info_beam, grow_info_beam_excl):
        """ Grow the rule by one step.
        
        Parameters
        ---
        grow_info_beam : GrowInfoBeam object
            Beam of rules with previously covered instances
        grow_info_beam_excl : GrowInfoBeam object
            Beam of rules without previously covered instances
            
        Returns
        ---
        None
        """
        pass

    def calculate_mdl_gain(self, bi_array, excl_bi_array, icol, cut_option):
        """ Calculate the MDL gain when adding a cut to the rule.
        
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
        pass

    def _print(self):
        """ Print the rule.
        
        Parameters
        ---
        None
            
        Returns
        ---
        readable : str
            String representation of the rule
        """
        pass