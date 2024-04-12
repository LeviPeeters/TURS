import numpy as np
import sys

import utils_calculating_cl
import nml_regret
import Beam
import DataInfo

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
                 ruleset,
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

        # The feature arrays take up far too much memory, especially because empty rules cover the entire dataset\
        # self.features = self.data_info.features[indices]
        self.target = self.data_info.target[indices]
        # self.features_excl = self.data_info.features[indices_excl]
        self.target_excl = self.data_info.target[indices_excl]


        # Condition matrix containing the rule literals and a boolean array to show which features have a condition
        self.condition_matrix = condition_matrix
        self.condition_bool = (~np.isnan(condition_matrix[0])).astype(int) + (~np.isnan(condition_matrix[1])).astype(int)

        # Probabilities and regrets
        self.prob_excl = self._calc_probs(target=self.target_excl)
        self.prob = self._calc_probs(target=self.target)
        self.regret_excl = nml_regret.regret(len(self.indices_excl), data_info.num_class)
        self.regret = nml_regret.regret(len(self.indices), data_info.num_class)
        self.negloglike_excl = utils_calculating_cl.calc_negloglike(p=self.prob_excl, n=len(self.indices_excl))
        self.cl_model = self.ruleset.model_encoding.rule_cl_model_dep(self.condition_matrix, col_orders=icols_in_order)

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
        if self.rule_base is None:
            # candidate_cuts_selector = (candidate_cuts[icol] < np.max(self.features_excl[:, icol])) & \
            #                           (candidate_cuts[icol] > np.min(self.features_excl[:, icol]))
            candidate_cuts_selector = (candidate_cuts[icol] < np.max(self.data_info.features[self.indices_excl, [icol]].flatten())) & \
                                      (candidate_cuts[icol] > np.min(self.data_info.features[self.indices_excl, [icol]].flatten()))
            candidate_cuts_icol = candidate_cuts[icol][candidate_cuts_selector]
            # print(candidate_cuts[icol])# < np.max(self.data_info.features[self.indices_excl, [icol]]))
            # breakpoint()
        else:
            # candidate_cuts_selector = (candidate_cuts[icol] < np.max(self.features[:, icol])) & \
            #                           (candidate_cuts[icol] > np.min(self.features[:, icol]))
            candidate_cuts_selector = (candidate_cuts[icol] < np.max(self.data_info.features[self.indices, [icol]])) & \
                                      (candidate_cuts[icol] > np.min(self.data_info.features[self.indices, [icol]]))
            candidate_cuts_icol = candidate_cuts[icol][candidate_cuts_selector]
        return candidate_cuts_icol

    def update_grow_beam(self, bi_array, excl_bi_array, icol, cut, cut_option, incl_coverage, excl_coverage,
                         grow_info_beam: Beam.GrowInfoBeam, incl_or_excl, _validity):
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
        if incl_or_excl == "incl":
            cov_percent = grow_info[f"coverage_incl"] / self.coverage
        else:
            cov_percent = grow_info[f"coverage_excl"] / self.coverage_excl

        # Update the beams if the grow step is valid
        if _validity[f"res_{incl_or_excl}"]:
            grow_info_beam.update(grow_info, grow_info[f"normalized_gain_{incl_or_excl}"], cov_percent)

    def grow(self, grow_info_beam, incl_or_excl):
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

        candidate_cuts = self.data_info.candidate_cuts

        # Consider each feature
        for icol in range(self.data_info.ncol):
            candidate_cuts_icol = self.get_candidate_cuts_icol_given_rule(candidate_cuts, icol)
            
            # Non-sparse
            # bi_array = self.data_info.features[:, icol]

            # Sparse: The binary array needs to be converted to dense and flattened, as sparse matrices do not reduce in dimension after slicing
            bi_array = self.data_info.features[:, [icol]].todense().flatten()
 


            # Consider every candidate cut point
            for i, cut in enumerate(candidate_cuts_icol):
                # if self.printing:
                #     print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - rule.grow. feature: {icol} / {self.data_info.ncol}, cut: {i} / {len(candidate_cuts_icol)}")
                
                # Construct binary arrays indicating which features fall on each side of the cut
                # excl_left_bi_array = (self.features_excl[:, icol] < cut)
                self.excl_left_bi_array = (bi_array[self.indices_excl] < cut)
                self.excl_right_bi_array = ~self.excl_left_bi_array
                # left_bi_array = (self.features[:, icol] < cut)
                self.left_bi_array = (bi_array[self.indices] < cut)
                self.right_bi_array = ~self.left_bi_array

                # Check validity and skip if not valid
                _validity = self.validity_check(icol=icol, cut=cut)

                if self.data_info.not_use_excl_:
                    _validity["res_excl"] = False

                if _validity["res_excl"] == False and _validity["res_incl"] == False:
                    continue

                incl_left_coverage, incl_right_coverage = np.count_nonzero(self.left_bi_array), np.count_nonzero(
                    self.right_bi_array)
                excl_left_coverage, excl_right_coverage = np.count_nonzero(self.excl_left_bi_array), np.count_nonzero(
                    self.excl_right_bi_array)

                # Question: Why is there no check on incl_coverage being 0?
                if excl_left_coverage == 0 or excl_right_coverage == 0:
                    continue

                # Update the beam with the results. We do this twice, because a cut can be < or >
                self.update_grow_beam(bi_array=self.left_bi_array, excl_bi_array=self.excl_left_bi_array, icol=icol,
                                      cut=cut, cut_option=constant.LEFT_CUT,
                                      incl_coverage=incl_left_coverage, excl_coverage=excl_left_coverage,
                                      grow_info_beam=grow_info_beam, incl_or_excl=incl_or_excl,
                                      _validity=_validity)

                self.update_grow_beam(bi_array=self.right_bi_array, excl_bi_array=self.excl_right_bi_array, icol=icol,
                                      cut=cut, cut_option=constant.RIGHT_CUT,
                                      incl_coverage=incl_right_coverage, excl_coverage=excl_right_coverage,
                                      grow_info_beam=grow_info_beam, incl_or_excl=incl_or_excl,
                                      _validity=_validity)

                del self.excl_left_bi_array, self.excl_right_bi_array, self.left_bi_array, self.right_bi_array


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

        cl_model = model_encoding.cl_model_after_growing_rule(rule=self, ruleset=self.ruleset, icol=icol,
                                                                      cut_option=cut_option)
        cl_data = data_encoding.get_cl_data_incl(self.ruleset, self, excl_bi_array=excl_bi_array, incl_bi_array=bi_array)
        cl_data_excl = data_encoding.get_cl_data_excl(self.ruleset, self, excl_bi_array)

        absolute_gain = self.ruleset.total_cl - cl_data - cl_model
        absolute_gain_excl = self.ruleset.total_cl - cl_data_excl - cl_model

        return {"cl_model": cl_model, "cl_data": cl_data, "cl_data_excl": cl_data_excl,
                "absolute_gain": absolute_gain, "absolute_gain_excl": absolute_gain_excl}

    def validity_check(self, icol, cut):
        """ Control function for validity check

        
        Parameters
        ---
        rule : Rule object
            Rule to be checked
        icol : int
            Index of the column for which the candidate cut is calculated
        cut : float
            Candidate cut
        
        Returns
        ---
        : dict
            contains the validity including and excluding overlapping instances
        """
        res_excl = True
        res_incl = True
        if self.data_info.alg_config.validity_check == "no_check":
            pass
        elif self.data_info.alg_config.validity_check == "excl_check":
            res_excl = self.check_split_validity_excl(icol, cut)
        elif self.data_info.alg_config.validity_check == "incl_check":
            res_incl = self.check_split_validity(icol, cut)
        elif self.data_info.alg_config.validity_check == "either":
            res_excl = self.check_split_validity_excl(icol, cut)
            res_incl = self.check_split_validity(icol, cut)
        else:
            # TODO: I should improve this error message a bit
            sys.exit("Error: the if-else statement should not end up here")
        return {"res_excl": res_excl, "res_incl": res_incl}

    def check_split_validity(self, icol, cut):
        """ Check validity when considering overlapping instances
        
        Parameters
        ---
        rule : Rule object
            Rule to be checked
        icol : int
            Index of the column for which the candidate cut is calculated
        cut : float
            Candidate cut
        
        Returns
        ---
        : bool
            Whether the split is valid
        """
        # indices_left, indices_right = rule.indices[rule.features[:, icol] < cut], rule.indices[rule.features[:, icol] >= cut]
        # indices_left, indices_right = self.indices[self.data_info.features[self.indices][:, icol] < cut], self.indices[self.data_info.features[self.indices][:, icol] >= cut]
        indices_left, indices_right = np.where(self.left_bi_array)[0], np.where(self.right_bi_array)[0]

        # Compute probabilities for the coverage of the rule and both sides of the split
        p_left = utils_calculating_cl.calc_probs(self.data_info.target[indices_left], self.data_info.num_class)
        p_right = utils_calculating_cl.calc_probs(self.data_info.target[indices_right], self.data_info.num_class)

        # Compute the negative log-likelihoods for these probabilities
        # Compute the negative log-likelihoods for these probabilities
        nll_rule = utils_calculating_cl.calc_negloglike(self.prob, self.coverage)
        nll_left = utils_calculating_cl.calc_negloglike(p_left, len(indices_left))
        nll_right = utils_calculating_cl.calc_negloglike(p_right, len(indices_right))

        # The extra model code length this split would add
        cl_model_extra = self.ruleset.model_encoding.cached_cl_model["l_cut"][0][icol]  # 0 represents for the "one split cut", instead of "two-splits cut"
        cl_model_extra += np.log2(self.ruleset.model_encoding.data_ncol_for_encoding)
        num_vars = np.sum(self.condition_bool > 0)
        cl_model_extra += self.ruleset.model_encoding.cached_cl_model["l_number_of_variables"][num_vars + 1] - \
                        self.ruleset.model_encoding.cached_cl_model["l_number_of_variables"][num_vars]

        # if this validity score is positive, the split decreases the total code length
        validity = nll_rule + nml_regret.regret(self.coverage, self.data_info.num_class) \
                    - nll_left - nml_regret.regret(len(indices_left), self.data_info.num_class) \
                    - nll_right - nml_regret.regret(len(indices_right), self.data_info.num_class) \
                    - cl_model_extra

        return (validity > 0)

        

    def check_split_validity_excl(self, icol, cut):
        """ Check validity without considering overlapping instances
        
        Parameters
        ---
        rule : Rule object
            Rule to be checked
        icol : int
            Index of the column for which the candidate cut is calculated
        cut : float
            Candidate cut
        
        Returns
        ---
        : bool
            Whether the split is valid
        """
        # indices_left, indices_right = rule.indices_excl[rule.features_excl[:, icol] < cut], rule.indices_excl[rule.features_excl[:, icol] >= cut]
        # indices_left, indices_right = self.indices_excl[self.data_info.features[self.indices_excl][:, icol] < cut], self.indices_excl[self.data_info.features[self.indices_excl][:, icol] >= cut]
        indices_left, indices_right = np.where(self.excl_left_bi_array)[0], np.where(self.excl_right_bi_array)[0]

        # Compute probabilities for the coverage of the rule and both sides of the split
        p_left = utils_calculating_cl.calc_probs(self.data_info.target[indices_left], self.data_info.num_class)
        p_right = utils_calculating_cl.calc_probs(self.data_info.target[indices_right], self.data_info.num_class)

        # Compute the negative log-likelihoods for these probabilities
        nll_rule = utils_calculating_cl.calc_negloglike(self.prob_excl, self.coverage_excl)
        nll_left = utils_calculating_cl.calc_negloglike(p_left, len(indices_left))
        nll_right = utils_calculating_cl.calc_negloglike(p_right, len(indices_right))

        # The extra model code length this split would add
        cl_model_extra = self.ruleset.model_encoding.cached_cl_model["l_cut"][0][icol]  # 0 represents for the "one split cut", instead of "two-splits cut"
        cl_model_extra += np.log2(self.ruleset.model_encoding.data_ncol_for_encoding)
        num_vars = np.sum(self.condition_bool > 0)
        cl_model_extra += self.ruleset.model_encoding.cached_cl_model["l_number_of_variables"][num_vars + 1] - \
                        self.ruleset.model_encoding.cached_cl_model["l_number_of_variables"][num_vars]

        # if this validity score is positive, the split decreases the total code length
        validity = nll_rule + nml_regret.regret(self.coverage_excl, self.data_info.num_class) \
                    - nll_left - nml_regret.regret(len(indices_left), self.data_info.num_class) \
                    - nll_right - nml_regret.regret(len(indices_right), self.data_info.num_class) \
                    - cl_model_extra

        return (validity > 0)

    def __str__(self):
        """ String representation of the rule for printing
        Should print categorical features in a more readable way
        
        Parameters
        ---
        None
            
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

        readable += "\n"

        readable += "Then:\n"
        if len(self.prob) > 5:
            readable += f"Highest probability is {max(self.prob)} for outcome {label_names[np.argmax(self.prob)]}"
        else:
            for i in range(len(label_names)):
                readable += f"Probability of {label_names[i]} is {round(self.prob[i], 2)}\n"
        readable += f"Coverage of this rule: {self.coverage}\n"

        return readable
