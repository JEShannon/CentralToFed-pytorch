from clientBase import fedClient
import torch
import torch.nn as nn

#This client is just a simple client that accepts weights via inputs, and sends results via returns.
#This is made for use with predictions that are simple binary data, like Breast Cancer Wisconsin.
class binaryClient(fedClient):

    def __init__(self, model, dataSet):
        super().__init__("binary client")
        self.model = model
        self.data = dataSet

    def train(self, weights, noiseFn=None, noise=0.0, sensitivity=1.0, learningRate=0.001, lossFn=nn.CrossEntropyLoss, optim=torch.optim.Adam):
        #### TO DO: Add checking to ensure the configuration (noise, model, etc) are valid, with errors to allow for debugging
        model = self.model()
        model.load_state_dict(weights)
        model.train()
        loss_fun = lossFn()
        optimizer = optim(model.parameters(), lr=learningRate)
        total = 0
        correct = 0
        net_loss = 0.0
        for data, labels in dataset:
            data = data.to(DEVICE)
            labels = labels.to(DEVICE)
            probs = model(data)
            loss = loss_fun(probs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            net_loss += loss.item() * data.size(0)
            #now generate accuracy and loss
            predicted = [1 if x.item() > 0.5 else 0 for x in probs]
            for i in range(len(predicted)):
                if(predicted[i] == labels[i].item()):
                  correct += 1
            total += labels.size(0)
        finalWeights = model.state_dict()
        #only perturb if we have a noise function and EITHER:
        # 1. Have a noise parameter
        # 2. Don't need a noise parameter
        #We still must check whether we actually use the noise parameter obviously, but it ensures the config is valid.
        if(noiseFn and (noise or (not noiseFn.useNoiseArg()))):
            if(noiseFn.useNoiseArg()):
                finalWeights = noiseFn.perturb(finalWeights, noise=noise, sensitivity=sensitivity)
            else:
                finalWeights = noiseFn.perturb(finalWeights, sensitivity=sensitivity)
        return finalWeights, netloss/total, correct/total
            
