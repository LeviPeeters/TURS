# Various functions for constraining the growth of a rule

import sys
import numpy as np
import utils_calculating_cl
import nml_regret

def validity_check(rule, icol, cut):
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
    if rule.data_info.alg_config.validity_check == "no_check":
        pass
    elif rule.data_info.alg_config.validity_check == "excl_check":
        res_excl = check_split_validity_excl(rule, icol, cut)
    elif rule.data_info.alg_config.validity_check == "incl_check":
        res_incl = check_split_validity(rule, icol, cut)
    elif rule.data_info.alg_config.validity_check == "either":
        res_excl = check_split_validity_excl(rule, icol, cut)
        res_incl = check_split_validity(rule, icol, cut)
    else:
        sys.exit("Error: the if-else statement should not end up here")
    return {"res_excl": res_excl, "res_incl": res_incl}

def check_split_validity(rule, icol, cut):
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
    indices_left, indices_right = rule.indices[rule.features[:, icol] < cut], rule.indices[rule.features[:, icol] >= cut]

    # Compute probabilities for the coverage of the rule and both sides of the split
    p_rule = rule.prob
    p_left = utils_calculating_cl.calc_probs(rule.data_info.target[indices_left], rule.data_info.num_class)
    p_right = utils_calculating_cl.calc_probs(rule.data_info.target[indices_right], rule.data_info.num_class)

    # Compute the negative log-likelihoods for these probabilities
    nll_rule = utils_calculating_cl.calc_negloglike(p_rule, rule.coverage)
    nll_left = utils_calculating_cl.calc_negloglike(p_left, len(indices_left))
    nll_right = utils_calculating_cl.calc_negloglike(p_right, len(indices_right))

    # The extra model code length this split would add
    cl_model_extra = rule.ruleset.model_encoding.cached_cl_model["l_cut"][0][icol]  # 0 represents for the "one split cut", instead of "two-splits cut"
    cl_model_extra += np.log2(rule.ruleset.model_encoding.data_ncol_for_encoding)
    num_vars = np.sum(rule.condition_bool > 0)
    cl_model_extra += rule.ruleset.model_encoding.cached_cl_model["l_number_of_variables"][num_vars + 1] - \
                      rule.ruleset.model_encoding.cached_cl_model["l_number_of_variables"][num_vars]

    # if this validity score is positive, the split decreases the total code length
    validity = nll_rule + nml_regret.regret(rule.coverage, rule.data_info.num_class) \
                - nll_left - nml_regret.regret(len(indices_left), rule.data_info.num_class) \
                - nll_right - nml_regret.regret(len(indices_right), rule.data_info.num_class) \
                - cl_model_extra

    return (validity > 0)

    

def check_split_validity_excl(rule, icol, cut):
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
    indices_left, indices_right = rule.indices_excl[rule.features_excl[:, icol] < cut], rule.indices_excl[rule.features_excl[:, icol] >= cut]

    # Compute probabilities for the coverage of the rule and both sides of the split
    p_rule = rule.prob_excl
    p_left = utils_calculating_cl.calc_probs(rule.data_info.target[indices_left], rule.data_info.num_class)
    p_right = utils_calculating_cl.calc_probs(rule.data_info.target[indices_right], rule.data_info.num_class)

    # Compute the negative log-likelihoods for these probabilities
    nll_rule = utils_calculating_cl.calc_negloglike(p_rule, rule.coverage_excl)
    nll_left = utils_calculating_cl.calc_negloglike(p_left, len(indices_left))
    nll_right = utils_calculating_cl.calc_negloglike(p_right, len(indices_right))

    # The extra model code length this split would add
    cl_model_extra = rule.ruleset.model_encoding.cached_cl_model["l_cut"][0][icol]  # 0 represents for the "one split cut", instead of "two-splits cut"
    cl_model_extra += np.log2(rule.ruleset.model_encoding.data_ncol_for_encoding)
    num_vars = np.sum(rule.condition_bool > 0)
    cl_model_extra += rule.ruleset.model_encoding.cached_cl_model["l_number_of_variables"][num_vars + 1] - \
                      rule.ruleset.model_encoding.cached_cl_model["l_number_of_variables"][num_vars]

    # if this validity score is positive, the split decreases the total code length
    validity = nll_rule + nml_regret.regret(rule.coverage_excl, rule.data_info.num_class) \
                - nll_left - nml_regret.regret(len(indices_left), rule.data_info.num_class) \
                - nll_right - nml_regret.regret(len(indices_right), rule.data_info.num_class) \
                - cl_model_extra

    return (validity > 0)