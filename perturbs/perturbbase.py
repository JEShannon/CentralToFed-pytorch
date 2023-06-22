class ModelPerturb():

    def __init__(self, perturbStr = "No Perturbations", noiseArg=False):
        self.__desc = perturbStr
        self.__noiseArg = noiseArg

    def getDesc(self):
        return self.__desc

    def useNoiseArg(self):
        return self.__noiseArg

    def perturb(self, model):
        #This is implemented in the subclasses
        return model
