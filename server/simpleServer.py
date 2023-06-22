import torch
from serverBase import serverBase

#This type of server is straight forward, and sequentially trains each client.
#It uses the trainTest function to run a specified number of epochs with optionally defined weights.
#With no weights it uses a new model.
#For larger or more complicated networks, use the file-passing server instead.
class simpleServer(serverBase):

    def __init__(self, model, datafn, clientFn, aggregator, numClients=10):
        super.__init__("simple server")
        self.__model = model
        data, test_set = datafn(numClients)
        self.__datasets = data
        self.__aggregator = aggregator
        self.__clients = []
        for i in range(numClients):
            #initialize the clients
            self.__clients.append(clientFn(model, data[i]))

    #implemented in subclasses, and params fit the needs of the subclass
    def __doRound(self, global_m, perturb, noise, sensitivity, lossFn, optim, learning, binary=False):
        #update each client, which at this point is a copy of the global model with an
        #individual data set
        global_w = global_m.state_dict()
        clientResults = []
        c_los = []
        c_acc = []
        for client in self.__clients:
            #train each client
            c_weights, train_l, train_a = client.train(global_w, noiseFn=perturb, noise=noise, sensitivity=sensitivity, learningRate=learning, lossFn=lossFn, optim=optim)
            clientResults.append(c_weights)
            c_los.append(train_l)
            c_acc.append(train_a)
        new_global = self.__aggregator.aggregate(clientResults)
        new_global_m = self.__model()
        new_global_m.load_state_dict(new_global)
        return None

    #implemented in subclasses, but it is recommended to keep to this signature
    #TODO: USE A CLASS/DICTIONARY TO HOLD THE CONFIGURATION
    def trainTest(self, epochs, perturb=None, noiseBudget=None, sensitivity = 0.0, budgetMult=1.0, lossFn=torch.nn.CrossEntropyLoss, optim=torch.optim.Adam, learning=0.001, binary=False):
        pass
