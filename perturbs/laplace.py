from .perturbbase import ModelPerturb
import torch
from torch.distributions.laplace import Laplace

class LaplaceNoise(ModelPerturb):

    def __init__(self, noise = 1.0):
        super().__init__("Laplace Constant")
        self.__noise = noise

    def perturb(self, weights, sensitivity = 1.0):
        for layer in weights.keys():
        #check if the layer is not a float type (ie, shouldn't be noised)
            if(not isinstance(weights[layer].type(), torch.FloatTensor)):
                continue
            #make a laplace distribution for the layer, then sample it.
            scalar = sensitivity/self.__noise
            scale = torch.full(weights[layer].size(), scalar)
            noisy = Laplace(weights[layer], scale)
            weights[layer] = noisy.sample()
        return weights

"""
This class is like the prior, but you must set the noise for every round.
Useful if you want to only put noise on some rounds, for instance.
"""
class LaplaceVariableNoise(ModelPerturb):

    def __init__(self):
        super().__init__("Laplace Variable", noiseArg=True)


    def perturb(self, weights, noise=1.0, sensitivity=1.0):
        if noise == 0:
            #don't add noise if it is 0
            return weights
        for layer in weights.keys():
        #check if the layer is not a float type (ie, shouldn't be noised)
            if(not isinstance(weights[layer].type(), torch.FloatTensor)):
                continue
            #make a laplace distribution for the layer, then sample it.
            scalar = sensitivity/noise
            scale = torch.full(weights[layer].size(), scalar)
            noisy = Laplace(weights[layer], scale)
            weights[layer] = noisy.sample()
        return weights
