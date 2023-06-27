import torch

from configurations import FedConfiguration
from models.MNISTCNN import MNISTCNN
from models.MNISTDNN import MNISTDNN
from models.BCWDNN import BCWDNN
from models.ShakespeareLSTM import ShhakeSpeareLSTM
from models.Resnet import Resnet101, Resnet152, Resnet34_CIFAR10, Resnet50_CIFAR10
from datasets.datasetBase import *
from aggregation.federatedAveraging import fedAvg
from clients.simpleClient import simpleClient
from perturbs.laplace import LaplaceNoise
from budgets.constant import ConstantBudget


def getMNIST_CNNConfig():
    config = FedConfiguration()
    config["model"] = MNISTCNN
    config["dataFn"] = makeMNISTData
    config["aggregator"] = fedAvg()
    # Clients are held as a list so you can set a percentage as malicious
    # Ensuring this results in the correct number of clients is the server's job.
    config["clientFn"] = [simpleClient]
    config["clientRatios"] = [1.0]
    # training parameters
    config["lossFn"] = torch.nn.CrossEntropyLoss
    config["optimizer"] = torch.optim.Adam
    config["learningRate"] = 0.001
    # perturbation parameters
    config["pertubation"] = None
    config["noiseSensitivity"] = 1.0
    config["budget"] = None
    config["budgetMultiplier"] = 1.0
    # model testing options
    config["binaryResult"] = False # Are the results from the system True/False?
    config["topkTesting"] = None # Should testing consider the top k results when considering if the network is correct?
    # Additional note: this value is None (if it isn't used) or an integer value, where the value is k
    return config
    

def getMNIST_DNNConfig():
    config = FedConfiguration()
    config["model"] = MNISTDNN
    config["dataFn"] = makeMNISTData
    config["aggregator"] = fedAvg()
    config["clientFn"] = [simpleClient]
    config["clientRatios"] = [1.0]
    # training parameters
    config["lossFn"] = torch.nn.CrossEntropyLoss
    config["optimizer"] = torch.optim.SGD
    config["learningRate"] = 0.01
    # perturbation parameters
    config["pertubation"] = None
    config["noiseSensitivity"] = 1.0
    config["budget"] = None
    config["budgetMultiplier"] = 1.0
    # model testing options
    config["binaryResult"] = False
    config["topkTesting"] = None 
    return config

def getBCWConfig():
    config = FedConfiguration()
    config["model"] = BCWDNN
    config["dataFn"] = makeBCWData
    config["aggregator"] = fedAvg()
    config["clientFn"] = [simpleClient]
    config["clientRatios"] = [1.0]
    # training parameters
    config["lossFn"] = torch.nn.MSELoss
    config["optimizer"] = torch.optim.SGD
    config["learningRate"] = 0.01
    # perturbation parameters
    config["pertubation"] = None
    config["noiseSensitivity"] = 1.0
    config["budget"] = None
    config["budgetMultiplier"] = 1.0
    # model testing options
    config["binaryResult"] = True
    config["topkTesting"] = None 
    return config

def getSpeareConfig():
    config = FedConfiguration()
    config["model"] = ShakeSpeareLSTM
    config["dataFn"] = makeSpeareData
    config["aggregator"] = fedAvg()
    config["clientFn"] = [simpleClient]
    config["clientRatios"] = [1.0]
    # training parameters
    config["lossFn"] = torch.nn.CrossEntropyLoss
    config["optimizer"] = torch.optim.SGD
    config["learningRate"] = 0.01
    # perturbation parameters
    config["pertubation"] = None
    config["noiseSensitivity"] = 1.0
    config["budget"] = None
    config["budgetMultiplier"] = 1.0
    # model testing options
    config["binaryResult"] = True
    config["topkTesting"] = None 
    return config

def getCIFAR_Res101Config():
    config = FedConfiguration()
    config["model"] = Resnet101
    config["dataFn"] = makeCIFAR100Data
    config["aggregator"] = fedAvg()
    config["clientFn"] = [simpleClient]
    config["clientRatios"] = [1.0]
    # training parameters
    config["lossFn"] = torch.nn.CrossEntropyLoss
    config["optimizer"] = torch.optim.Adam
    config["learningRate"] = 0.001
    # perturbation parameters
    config["pertubation"] = None
    config["noiseSensitivity"] = 1.0
    config["budget"] = None
    config["budgetMultiplier"] = 1.0
    # model testing options
    config["binaryResult"] = True
    config["topkTesting"] = None 
    return config

def getCIFAR_Res152Config():
    config = FedConfiguration()
    config["model"] = Resnet152
    config["dataFn"] = makeCIFAR100Data
    config["aggregator"] = fedAvg()
    config["clientFn"] = [simpleClient]
    config["clientRatios"] = [1.0]
    # training parameters
    config["lossFn"] = torch.nn.CrossEntropyLoss
    config["optimizer"] = torch.optim.Adam
    config["learningRate"] = 0.001
    # perturbation parameters
    config["pertubation"] = None
    config["noiseSensitivity"] = 1.0
    config["budget"] = None
    config["budgetMultiplier"] = 1.0
    # model testing options
    config["binaryResult"] = True
    config["topkTesting"] = None 
    return config

def getCIFAR10_Res34Config():
    config = FedConfiguration()
    config["model"] = Resnet34_CIFAR10
    config["dataFn"] = makeCIFAR10Data
    config["aggregator"] = fedAvg()
    config["clientFn"] = [simpleClient]
    config["clientRatios"] = [1.0]
    # training parameters
    config["lossFn"] = torch.nn.CrossEntropyLoss
    config["optimizer"] = torch.optim.Adam
    config["learningRate"] = 0.001
    # perturbation parameters
    config["pertubation"] = None
    config["noiseSensitivity"] = 1.0
    config["budget"] = None
    config["budgetMultiplier"] = 1.0
    # model testing options
    config["binaryResult"] = True
    config["topkTesting"] = None 
    return config

def getCIFAR10_Res50Config():
    config = FedConfiguration()
    config["model"] = Resnet52_CIFAR10
    config["dataFn"] = makeCIFAR10Data
    config["aggregator"] = fedAvg()
    config["clientFn"] = [simpleClient]
    config["clientRatios"] = [1.0]
    # training parameters
    config["lossFn"] = torch.nn.CrossEntropyLoss
    config["optimizer"] = torch.optim.Adam
    config["learningRate"] = 0.001
    # perturbation parameters
    config["pertubation"] = None
    config["noiseSensitivity"] = 1.0
    config["budget"] = None
    config["budgetMultiplier"] = 1.0
    # model testing options
    config["binaryResult"] = True
    config["topkTesting"] = None 
    return config
