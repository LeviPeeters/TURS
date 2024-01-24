import numpy as np

import utils_calculating_cl
import nml_regret
import utils_readable
import Beam
import RuleGrowConstraint
import DataInfo
import Ruleset

import constant

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
                 data_info: DataInfo.DataInfo,
                 rule_base, 
                 condition_matrix, 
                 ruleset: Ruleset.Ruleset,
                 mdl_gain, 
                 mdl_gain_excl, 
                 icols_in_order):
        
        self.ruleset = ruleset
        self.data_info = data_info # Metadata
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

        # Feature- and target subvector covered by the rule
        self.features = self.data_info.features[indices]
        self.target = self.data_info.target[indices]
        self.features_excl = self.data_info.features[indices_excl]
        self.target_excl = self.data_info.target[indices_excl]

        # Condition matrix containing the rule literals and a boolean array to show which features have a condition
        self.condition_matrix = condition_matrix
        self.condition_bool = (~np.isnan(condition_matrix[0])).astype(int) + (~np.isnan(condition_matrix[1])).astype(int)

        # Probabilities and regrets
        self.prob_excl = self._calc_probs(target=self.target_excl_overlap)
        self.prob = self._calc_probs(target=self.target)
        self.regret_excl = nml_regret.regret(self.nrow_excl, data_info.num_class)
        self.regret = nml_regret.regret(self.nrow, data_info.num_class)
        self.negloglike_excl = utils_calculating_cl.calc_negloglike(p=self.prob_excl, n=len(self.indices_excl))
        self.cl_model = self.ruleset.model_encoding.rule_cl_model_dep(self.condition_matrix, col_orders=icols_in_order)

        self.mdl_gain = mdl_gain
        self.mdl_gain_excl = mdl_gain_excl


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
        return utils_calculating_cl.calc_probs(target, num_class=self.data_info.num_class, smoothed=False)

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
        if self.rule_base is None:
            candidate_cuts_selector = (candidate_cuts[icol] < np.max(self.features_excl[:, icol])) & \
                                      (candidate_cuts[icol] > np.min(self.features_excl[:, icol]))
            candidate_cuts_icol = candidate_cuts[icol][candidate_cuts_selector]
        else:
            candidate_cuts_selector = (candidate_cuts[icol] < np.max(self.features[:, icol])) & \
                                      (candidate_cuts[icol] > np.min(self.features[:, icol]))
            candidate_cuts_icol = candidate_cuts[icol][candidate_cuts_selector]
        return candidate_cuts_icol

    def update_grow_beam(self, bi_array, excl_bi_array, icol, cut, cut_option, incl_coverage, excl_coverage,
                         grow_info_beam: Beam.GrowInfoBeam, grow_info_beam_excl: Beam.GrowInfoBeam, _validity):
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

        Returns
        ---
        None
        """
        # Calculate the MDL gain
        info_theo_scores = self.calculate_mdl_gain(bi_array=bi_array, excl_bi_array=excl_bi_array,
                                                   icol=icol, cut_option=cut_option)
        
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
        excl_cov_percent = grow_info["coverage_excl"] / self.coverage_excl
        incl_cov_percent = grow_info["coverage_incl"] / self.coverage

        # Update the beams if the grow step is valid
        if _validity["res_excl"]:
            grow_info_beam_excl.update(grow_info, grow_info["normalized_gain_excl"], excl_cov_percent)

        if _validity["res_incl"]:
            grow_info_beam.update(grow_info, grow_info["normalized_gain_incl"], incl_cov_percent)

    def grow(self, grow_info_beam, grow_info_beam_excl):
        """ Grow the rule by one step and update the 
        
        Parameters
        ---
        grow_info_beam : DiverseCovarianceBeam object
            Beam object to store information for this grow step
        grow_info_beam_excl : DiverseCovarianceBeam object
            Beam of object to store information for this grow step without considering previously covered instances
            
        Returns
        ---
        None
        """
        candidate_cuts = self.data_info.candidate_cuts

        # Consider each feature
        for icol in range(self.data_info.ncol):
            candidate_cuts_icol = self.get_candidate_cuts_icol_given_rule(candidate_cuts, icol)

            # Consider every candidate cut point
            for i, cut in enumerate(candidate_cuts_icol):
                
                # Check validity and skip if not valid
                _validity = RuleGrowConstraint.validity_check(rule=self, icol=icol, cut=cut)
                if self.data_info.not_use_excl_:
                    _validity["res_excl"] = False

                if _validity["res_excl"] == False and _validity["res_incl"] == False:
                    continue

                # Construct binary arrays indicating which features fall on each side of the cut
                excl_left_bi_array = (self.features_excl[:, icol] < cut)
                excl_right_bi_array = ~excl_left_bi_array
                left_bi_array = (self.features[:, icol] < cut)
                right_bi_array = ~left_bi_array

                incl_left_coverage, incl_right_coverage = np.count_nonzero(left_bi_array), np.count_nonzero(
                    right_bi_array)
                excl_left_coverage, excl_right_coverage = np.count_nonzero(excl_left_bi_array), np.count_nonzero(
                    excl_right_bi_array)

                # QUESTION: Why is there no check on incl_coverage being 0?
                if excl_left_coverage == 0 or excl_right_coverage == 0:
                    continue

                # Update the beam with the results. We do this twice, because a cut can be < or >
                self.update_grow_beam(bi_array=left_bi_array, excl_bi_array=excl_left_bi_array, icol=icol,
                                      cut=cut, cut_option=constant.LEFT_CUT,
                                      incl_coverage=incl_left_coverage, excl_coverage=excl_left_coverage,
                                      grow_info_beam=grow_info_beam, grow_info_beam_excl=grow_info_beam_excl,
                                      _validity=_validity)

                self.update_grow_beam(bi_array=right_bi_array, excl_bi_array=excl_right_bi_array, icol=icol,
                                      cut=cut, cut_option=constant.RIGHT_CUT,
                                      incl_coverage=incl_right_coverage, excl_coverage=excl_right_coverage,
                                      grow_info_beam=grow_info_beam, grow_info_beam_excl=grow_info_beam_excl,
                                      _validity=_validity)


    def calculate_mdl_gain(self, bi_array, excl_bi_array, icol, cut_option):
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
        data_encoding, model_encoding = self.ruleset.data_encoding, self.ruleset.model_encoding

        cl_model = model_encoding.cl_model_after_growing_rule_on_icol(rule=self, ruleset=self.ruleset, icol=icol,
                                                                      cut_option=cut_option)
        cl_data = data_encoding.get_cl_data_incl(self.ruleset, self, excl_bi_array=excl_bi_array, incl_bi_array=bi_array)
        cl_data_excl = data_encoding.get_cl_data_excl(self.ruleset, self, excl_bi_array)

        absolute_gain = self.ruleset.total_cl - cl_data - cl_model
        absolute_gain_excl = self.ruleset.total_cl - cl_data_excl - cl_model

        return {"cl_model": cl_model, "cl_data": cl_data, "cl_data_excl": cl_data_excl,
                "absolute_gain": absolute_gain, "absolute_gain_excl": absolute_gain_excl}

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