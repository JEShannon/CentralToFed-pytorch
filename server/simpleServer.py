import torch
from torch.utils.data import DataLoader
from .serverBase import serverBase

#This type of server is straight forward, and sequentially trains each client.
#It uses the trainTest function to run a specified number of epochs with optionally defined weights.
#With no weights it uses a new model.
#For larger or more complicated networks, use the file-passing server instead.
class simpleServer(serverBase):

    def __init__(self, config):
        super().__init__("simple server")
        self.__config = config
        numClients = 10   #### TO DO:  Add a value to track this value instead
        self.__model = config["model"]
        data, testset = config["dataFn"](numClients)
        self.__datasets = data
        test_set = [next(iter(DataLoader(testset, batch_size=500, shuffle=True)))]
        self.__testSet = test_set
        self.__aggregator = config["aggregator"]
        self.__clients = []
        for i in range(numClients):
            #initialize the clients
            #### TODO: Use the config's clients array to have different clients in the same system -- NEEDED for scenarios with malicious clients.
            self.__clients.append(config["clientFn"][0](self.__model, data[i]))

    def test(self, weights):
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        testModel = self.__model()
        testModel.load_state_dict(weights)
        testModel = testModel.to(DEVICE)
        testModel.eval()
        loss = 0.0
        total = 0
        correct = 0
        runs = 0.0
        for data, labels in self.__testSet:
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            probs = testModel(data)
            loss_fun = self.__config["lossFn"]()
            loss += loss_fun(probs, labels).item() * labels.size(0)
            #print(labels.shape)
            #print(labels)
            if(self.__config["binaryResult"]):
              #print('Binary!')
              predicted = [1 if x.item() > 0.5 else 0 for x in probs]
              #print(predicted[:8], labels[:8])
              #print([(0 if predicted[x] < 0.5 else 1 ) == labels[x].item() for x in range(8)])
              for i in range(len(predicted)):
                if(predicted[i] == labels[i].item()):
                  correct += 1
              total += labels.size(0)
              #print(total)
              continue
            if(self.__config["topkTesting"]):
              _, predicted = torch.topk(probs.data, top, 1)
              #print(predicted.shape)
              #print(predicted)
            else:
              _, predicted = torch.max(probs.data, 1)
            total += labels.size(0)
            if(self.__config["topkTesting"]):
              for x in range(predicted.size(dim=0)):
                #print(labels[x])
                #print(predicted[x])
                if(labels[x] in predicted[x]):
                  #print(True)
                  correct += 1
            else:
              correct += (predicted == labels).sum().item()
            runs += 1.0

        #print(loss/total, correct/total, ones/total)
        return loss/total, correct/total

    #implemented in subclasses, and params fit the needs of the subclass
    def __doRound(self, global_w, noise):
        #update each client, which at this point is a copy of the global model with an
        #individual data set
        clientResults = []
        c_los = [] # client losses
        c_acc = [] # client accuracies
        for client in self.__clients:
            #train each client - For the simple server this is done sequentially
            c_weights, train_l, train_a = client.train(global_w, noiseFn=self.__config["perturbation"], noise=noise, sensitivity=self.__config["noiseSensitivity"],
                                                       learningRate=self.__config["learningRate"], lossFn=self.__config["lossFn"], optim=self.__config["optimizer"])
            clientResults.append(c_weights)
            c_los.append(train_l)
            c_acc.append(train_a)
        new_global = self.__aggregator.aggregate(clientResults)
        return new_global, c_los, c_acc

    #implemented in subclasses, but it is recommended to keep to this signature
    #TODO: BETTER COMMUNICATE TRAINING OUTPUTS (LIKE CURRENT EPOCH COUNT, ETC)
    def trainTest(self, weights=None, epochs=10):
        #for testing, this pulls just a small sample of the testing dataset
        test_loader = DataLoader(self.__testSet, batch_size=500, shuffle=True)
        test_set = [next(iter(test_loader))]
        noiseBudget = self.__config["budget"]

        train_acc = []
        train_loss = []
        test_acc = []
        test_loss = []

        #initialize the model weights, if needed
        if(not weights):
            weights = self.__model().state_dict()

        total_budget = 0.0

        earlyStop = False
        
        #the following values are for early stopping
        tol = 0.00025
        inter = 25
        curr = 200.0
        count = 0
        hasStopped = False

        #begin training for the specified number of epochs
        for epoch in range(epochs):
          print('Round',epoch+1)
          if(noiseBudget):
            budget = noiseBudget.getBudgetAt(epoch)
          else:
            budget = 0.0
          total_budget += budget
          #print("Training clients!")
          weights, loss, acc = self.__doRound(weights, budget)
          train_acc.append(acc)
          train_loss.append(loss)
          g_loss, g_acc = self.test(weights)
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
          if(self.__config["topkTesting"]):
            k_loss,k_acc = self.test(weights)
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
        print('total budget:', (None if total_budget == 0 else total_budget))
        return weights, train_acc, train_loss, test_acc, test_loss, total_budget
