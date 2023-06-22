import torch.nn as nn

#This is purely to act as a framework that other models are based in.
#At the time of writing, this is a wrapper for a pytorch model with some extra info to denote it is malicious or not.

class CentralToFedBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.malicious = False

    #This private method is usually set at network creation.  If you want to change it on the fly, implement it in the subclass.
    def _setHostile(self, value):
        self.malicious = value

    def forward(self, inputs):
        pass
