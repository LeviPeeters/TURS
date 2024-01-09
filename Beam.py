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
        """ Update the beam with a new growth step, unless a growth step with the same coverage is already in the beam.

        Parameters
        ---
        info : Dict
            Information about the growth step
        gain : float
            Gain of the growth step
        """
        pass


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
        """ Update the beam with a new growth step if the new literal is diverse enough

        Parameters
        ---
        info : Dict
            Information of the growth step
        gain : float
            Gain of the growth step
        coverage_percentage : float
            Coverage percentage of the growth step
        """
        pass