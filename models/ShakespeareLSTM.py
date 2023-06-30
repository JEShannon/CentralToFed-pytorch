import torch
import torch.nn as nn
from .modelBase import CentralToFedBase

#based on the model used in the following paper.
#https://ieeexplore.ieee.org/abstract/document/9183378

class SpeareLSTM(CentralToFedBase):
    def __init__(self):
        super().__init__()
        self.n_characters = 100
        self.hidden_dim = 100
        self.n_layers = 2
        self.len_seq = 80
        self.batch_size = 100
        self.embed_dim = 8

        self.embed = nn.Embedding(self.n_characters, self.embed_dim)

        self.lstm = nn.LSTM(
            self.embed_dim, self.hidden_dim, self.n_layers, batch_first=True
        )

        self.fc = nn.Linear(self.hidden_dim, self.n_characters)

    def forward(self, x):

        embed_x = self.embed(x)
        output, _ = self.lstm(embed_x)
        output = self.fc(output[:, -1])
        return output

    def init_hidden(self, batch_size):

        self.hidden = (
            torch.zeros(self.n_layers, batch_size, self.hidden_dim),
            torch.zeros(self.n_layers, batch_size, self.hidden_dim),
        )

def ShakeSpeareLSTM():
    return SpeareLSTM()
