import torch

class serverBase():

    def __init__(self, desc):
        self.__desc = desc

    def getDesc(self)::
        return self.__desc

    #implemented in subclasses, and params fit the needs of the subclass
    def doRound(self):
        return None

    #implemented in subclasses, but it is recommended to keep to this signature
    #TODO: USE A CLASS TO HOLD THE CONFIGURATION
    def trainTest(self, numUsers, epochs, noiseBudget=None, budgetMult=1.0, datafn, modelfn, lossFn=torch.nn.CrossEntropyLoss, optim=torch.optim.Adam, learning=0.001, binary=False):
        pass
