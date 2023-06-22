import torch.nn as nn
from modelBase import CentralToFedBase

#A simple model

class MNIST_DNN(CentraloFedBase):
  def __init__(self):
        super(FedLap, self).__init__()
        
        self.layer1 = nn.Linear(784, 800)
        
        self.layer2 = nn.Linear(800, 10)

  def forward(self, x):

        x = x.reshape((len(x), -1))
        #x = x.reshape((-1, 784))

        print(x.shape)

        x = self.layer1(x)
        x = self.layer2(x)
        return x

def MNISTDNN():
    return MNIST_DNN()
