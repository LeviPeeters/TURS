import numpy as np

class GrowInfoBeam:
    """
    This class is used to store the information of a grow step in the beam search.
    It stores the information of the grow step, the gain of the grow step, and the coverage of the grow step.
    Specifically, we check exclude the growth result if some other growth result with the same "cover" is already in the beam.
    """

    def __init__(self, width):
        self.infos = []
        self.gains = []

        self.worst_gain = None
        self.which_worst_gain = None
        self.width = width

        self.coverage_list = [] 

    def update(self, info, gain):
        """ Update the beam with a new growth step. 
        The step is only added if it improves the step with worst gain, or if there is less than self.width steps in the beam.
        Additionally, the new step is checked for diversity from previous steps.

        Parameters
        ---
        info : Dict
            Information about the growth step
        gain : float
            Gain of the growth step
        """
        info_coverage = np.count_nonzero(info["incl_bi_array"])
        skip_flag = False

        # Check if a growth step with the same coverage is already in the beam
        # TODO: Wouldn't this still allow steps with very similar coverage to be added?
        if info_coverage in self.coverage_list:
            which_equal = self.coverage_list.index(info_coverage)
            bi_array_in_list = self.infos[which_equal]["incl_bi_array"]
            bi_array_input = info["incl_bi_array"]
            if np.array_equal(bi_array_in_list, bi_array_input):
                skip_flag = True

        if skip_flag is False:
            # If there are less steps in the beam than the width, add the new step
            if len(self.infos) < self.width:
                self.infos.append(info)
                self.gains.append(gain)
                self.worst_gain = np.min(self.gains)
                self.whichworst_gain = np.argmin(self.gains)

                self.coverage_list.append(info_coverage)
            
            # If there are already self.width steps in the beam, replace the step with worst gain if the new step has a better gain
            else:
                if gain > self.worst_gain:
                    self.gains.pop(self.whichworst_gain)
                    self.infos.pop(self.whichworst_gain)
                    self.infos.append(info)
                    self.gains.append(gain)
                    self.worst_gain = np.min(self.gains)
                    self.whichworst_gain = np.argmin(self.gains)

                    self.coverage_list.pop(self.whichworst_gain)
                    self.coverage_list.append(info_coverage)


class DiverseCovBeam:
    """
    This class is used to store the information of a grow step in the beam search, with the coverage percentage as the key.
    We use it to control the diversity of the coverage of the growth results in the beam.
    """
    def __init__(self, width):
        self.coverage_percentage = np.linspace(0, 1, width + 1)[1:]
        self.infos = {}
        for i, cov in enumerate(self.coverage_percentage):
            self.infos[i] = None

        self.gains = {}
        for i, cov in enumerate(self.coverage_percentage):
            self.gains[i] = None

        self.worst_gain = None
        self.width = width

    def update(self, info, gain, coverage_percentage):
        """ Update the beam with a new growth step. 
        
        
        Parameters
        ---
        info : Dict
            Information of the growth step
        gain : float
            Gain of the growth step
        coverage_percentage : float
            Coverage percentage of the growth step
        """
        # Find which coverage interval the new step belongs to
        which_coverage_interval = np.searchsorted(a=self.coverage_percentage, v=coverage_percentage)
        
        # If this interval is empty, add the new step
        if self.infos[which_coverage_interval] is None:
            self.infos[which_coverage_interval] = info
            self.gains[which_coverage_interval] = gain
            self.worst_gain = np.min(self.gains)
        
        # Otherwise, add the new step if it has a better gain than the step with worst gain in the interval
        else:
            skip_flag = False
            info_coverage = np.count_nonzero(info["incl_bi_array"])
            if info_coverage == np.count_nonzero(self.infos[which_coverage_interval]["incl_bi_array"]):
                if np.array_equal(info["incl_bi_array"], self.infos[which_coverage_interval]["incl_bi_array"]):
                    skip_flag = True
            if not skip_flag and gain > self.gains[which_coverage_interval]:
                self.infos[which_coverage_interval] = info
                self.gains[which_coverage_interval] = gain
                self.worst_gain = np.min(self.gains)