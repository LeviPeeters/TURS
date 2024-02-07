import numpy as np
import utils_calculating_cl

def get_rule_local_prediction_for_unseen_data_this_rule_only(rule, X_test, y_test):
    """ This function computes the local prediction for a rule, given a test set.
    
    Parameters
    ----------
    rule : Rule
        The rule for which we want to compute the local prediction.
    X_test : np.array
        The test set.
    y_test : np.array
        The test labels.
    
    Returns
    -------
     : List
        Probability distributions and coverage for the prediction
    """

def get_rule_local_prediction_for_unseen_data(ruleset, X_test, y_test):
    """ This function computes the local prediction for a ruleset, given a test set.
    
    Parameters
    ----------
    ruleset : RuleSet
        The ruleset for which we want to compute the local prediction.
    X_test : np.array
        The test set.
    y_test : np.array
        The test labels.
    
    Returns
    -------
     : Dict
        Probability distributions and coverages for the prediction
    """

def predict_ruleset(ruleset, X_test, y_test):
    """ This function computes the local prediction using a ruleset, given a test set.
    
    Parameters
    ----------
    ruleset : RuleSet
        The ruleset for which we want to compute the local prediction.
    X_test : np.array
        The test set.
    y_test : np.array
        The test labels.
    
    Returns
    -------
     : Array
        Probability distributions for the prediction
    """
    if type(X_test) != np.ndarray:
        X_test = X_test.to_numpy()

    if type(y_test) != np.ndarray:
        y_test = y_test.to_numpy().flatten()

    prob_predicted = np.zeros((len(X_test), ruleset.data_info.num_class), dtype=float)
    cover_matrix = np.zeros((len(X_test), len(ruleset.rules) + 1), dtype=bool)

    test_uncovered_bool = np.ones(len(X_test), dtype=bool)
    for ir, rule in enumerate(ruleset.rules):
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
        bool_model = np.zeros(len(ruleset.data_info.target), dtype=bool)
        for i_tt, tt in enumerate(t):
            if tt == 1:
                if i_tt == len(ruleset.rules):
                    bool_model = ruleset.uncovered_bool
                else:
                    bool_model = np.bitwise_or(bool_model, ruleset.rules[i_tt].bool_array)
        unique_id_prob_dir[z] = utils_calculating_cl.calc_probs(ruleset.data_info.target[bool_model],
                                           ruleset.data_info.num_class)

    for i in range(len(prob_predicted)):
        prob_predicted[i] = unique_id_prob_dir[unique_id[i]]

    return prob_predicted
