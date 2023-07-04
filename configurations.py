from perturbs.laplace import LaplaceNoise
from budgets.budgetbase import budget
from budgets.constant import ConstantBudget

"""
This class exists to hold all the options for the server training.

If you want to add your own options either use the extra options,
or just make a subclass with the options added in.

Currently, this is functionally a wrapper on a dictionary, but I have it in case I need to add more functionality later.
"""
class FedConfiguration():

    def __init__(self):
        self.__options = {}

    def __getitem__(self, key):
        return self.__options.get(key)

    def __setitem__(self, key, value):
        self.__options[key] = value

    def hasNoise(self):
        return self["perturbation"] and (self["budget"] or (not self["perturbation"].useNoiseArg()))

    def usePerturbation(self, method=LaplaceNoise(), trainingBudget=1.0, sensitivity=1.0):
        if(isinstance(trainingBudget, budget)):
            self["budget"] = trainingBudget
        elif(isinstance(trainingBudget, float)):
            self["budget"] = ConstantBudget(trainingBudget)
        else:
            self["budget"] = ConstantBudget((float) (trainingBudget))
        self["pertubation"] = method
        self["noiseSensitivity"] = sensitivity
