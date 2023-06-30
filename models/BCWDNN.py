import torch
import torch.nn as nn
from .modelBase import CentralToFedBase

#based on the model used in the following paper.
#https://proceedings.mlr.press/v130/fraboni21a.html

class BreastCancerDNN(CentralToFedBase):
  def __init__(self):
    super().__init__()

    #all layers use Glorot initialization to avoid training difficulties.
    self.dnn1 = nn.Linear(30,30)
    nn.init.xavier_normal_(self.dnn1.weight)

    self.dnn2 = nn.Linear(30,15)
    nn.init.xavier_normal_(self.dnn2.weight)

    self.dnn3 = nn.Linear(15,7)
    nn.init.xavier_normal_(self.dnn3.weight)

    self.dnn4 = nn.Linear(7,3)
    nn.init.xavier_normal_(self.dnn4.weight)

    self.dnn5 = nn.Linear(3,1)
    nn.init.xavier_normal_(self.dnn5.weight)

  def forward(self, x):
    x = self.dnn1(x)
    x = self.dnn2(x)
    x = self.dnn3(x)
    x = self.dnn4(x)
    x = torch.round(torch.sigmoid(self.dnn5(x)))
    return x

def BCWDNN():
  return BreastCancerDNN()
