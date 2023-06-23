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
        self.__testSet = test_set
        self.__aggregator = aggregator
        self.__clients = []
        for i in range(numClients):
            #initialize the clients
            self.__clients.append(clientFn(model, data[i]))

    def test

    #implemented in subclasses, and params fit the needs of the subclass
    def __doRound(self, global_w, perturb, noise, sensitivity, lossFn, optim, learning, binary=False):
        #update each client, which at this point is a copy of the global model with an
        #individual data set
        clientResults = []
        c_los = [] # client losses
        c_acc = [] # client accuracies
        for client in self.__clients:
            #train each client - For the simple server this is done sequentially
            c_weights, train_l, train_a = client.train(global_w, noiseFn=perturb, noise=noise, sensitivity=sensitivity, learningRate=learning, lossFn=lossFn, optim=optim)
            clientResults.append(c_weights)
            c_los.append(train_l)
            c_acc.append(train_a)
        new_global = self.__aggregator.aggregate(clientResults)
        return new_global, c_los, c_acc

    #implemented in subclasses, but it is recommended to keep to this signature
    #TODO: USE A CLASS/DICTIONARY TO HOLD THE CONFIGURATION
    #TODO: BETTER COMMUNICATE TRAINING OUTPUTS (LIKE CURRENT EPOCH COUNT, ETC)
    def trainTest(self, weights=None, epochs, perturb=None, noiseBudget=None, sensitivity = 1.0, budgetMult=1.0, lossFn=torch.nn.CrossEntropyLoss, optim=torch.optim.Adam, learning=0.001, binary=False, topk=None):
        #for testing, this pulls just a small sample of the testing dataset
        test_loader = DataLoader(self.__testSet, batch_size=500, shuffle=True)
        test_set = [next(iter(test_loader))]

        train_acc = []
        train_loss = []
        test_acc = []
        test_loss = []

        #initialize the model weights, if needed
        if(not weights):
            weights = self.__model().state_dict()

        total_budget = 0.0
        
        #the following values are for early stopping
        tol = 0.00025
        inter = 25
        curr = 200.0
        count = 0
        hasStopped = False

        #begin training for the specified number of epochs
        for epoch in range(epochs):
          #print('Round',epoch+1)
          budget = noiseBudget.getBudgetAt(epoch)
          total_budget += budget
          #print("Training clients!")
          weights, loss, acc = self.__doRound(weights, peturb, budget, sensitivity=sensitivity, lossFn=lossFn, optim=optim, learning=learning, binary=binary)
          train_acc.append(acc)
          train_loss.append(loss)
          g_loss, g_acc = 0.0, 0.0 #test(model, test_set, lossfn=lossfn, binary=binary)  #testing function is not yet implemented #### TODO ####
          #print(g_loss, g_acc)
          #print("Testing!")
          if (abs(curr - g_acc) < tol) and not curr == 200.0 and earlyStop and not hasStopped:
            #increment up!
            count += 1
            if not (count < inter):
              print("Early stop at", epoch, g_acc)
              hasStopped = True
          elif(earlyStop and not hasStopped):
            #print(count)
            count = 0
            curr = g_acc
          if(topk):
            k_loss,k_acc = 0.0, 0.0 #test(model, test_set, topk, lossfn=lossfn, binary=binary)  #testing function is not yet implemented #### TODO ####
            test_acc.append(tuple([g_acc, k_acc]))
            test_loss.append(tuple([g_loss, k_loss]))
            print(test_acc)
            print(test_loss)
          else:
            test_acc.append(g_acc)
            test_loss.append(g_loss)
          if(hasStopped):
            break
        #print(test_acc[-1])
        print('total budget:', total_budget)
        return weights, train_acc, train_loss, test_acc, test_loss, total_budget
