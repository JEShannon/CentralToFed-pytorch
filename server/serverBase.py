import torch

class serverBase():

    def __init__(self, desc):
        self.__desc = desc

    def getDesc(self):
        return self.__desc

    #implemented in subclasses, and params fit the needs of the subclass
    def doRound(self):
        return None

    #implemented in subclasses, but it is recommended to keep to this signature
    def trainTest(self, weights, epochs):
        pass
