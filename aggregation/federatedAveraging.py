import torch
import copy
from .aggregateBase import aggregator

class fedAvg(aggregator):

    def __init__(self):
        super().__init__("FedAvg")

    def aggregate(self, clients):
        average_weights = copy.deepcopy(clients[0])
        for layer in average_weights.keys():
          if(not isinstance(average_weights[layer].type(), torch.FloatTensor)):
            # Don't average non-float values, as those aren't normally trainable values.
            continue
          for i in range(1, len(clients)):
            average_weights[layer] += clients[i][layer]
          average_weights[layer] = torch.div(average_weights[layer], len(clients))
        return average_weights
