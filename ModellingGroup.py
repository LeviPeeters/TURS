import numpy as np

class ModellingGroup:
    """
    A ModellingGroup object is a group of rules that are used for modelling, which we specifically use
    for handling the "overlap" among rules. When two rules overlap, we use the "union" to model the "intersection",
    which is the key for only allowing rules with similar probabilistic outputs to overlap.
    """
    def __init__(self, data_info, bool_cover, bool_use_for_model, rules_involved, prob_model, prob_cover):
        self.data_info = data_info
        self.bool_cover = bool_cover  # intersection of rules
        self.bool_model = bool_use_for_model  # union of rules
        self.rules_involvde = rules_involved

        self.cover_count = np.count_nonzero(bool_cover)
        self.use_for_model_count = np.count_nonzero(bool_use_for_model)

        self.prob_model = prob_model
        self.prob_cover = prob_cover
        self.negloglike = -self.cover_count * np.sum(prob_cover[prob_model != 0] * np.log2(prob_model[prob_model != 0]))

    def evaluate_rule_with_no_updating(self, indices):
        """ Approximately evaluate the neg_log_likelihood when growing the rule
        TODO: unclear what the nonRule part does

        Parameters
        ---
        indices : Array
            Indices after growth if we were to grow this rule

        Returns
        ---
        : float
            TODO
        """

    def evaluate_rule(self, rule, update_rule_index):
        """ calculate the negloglikelihood for the rule_and_mg and nonRule_and_mg;
        TODO: unclear what the nonRule part does
        
        Parameters
        ---
        rule : Rule object
            Rule to be added
        update_rule_index : int
            Index of the rule to be added

        Returns
        ---
        : float
            TODO
        """