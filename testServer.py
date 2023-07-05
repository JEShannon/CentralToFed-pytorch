from demoConfigs import getMNIST_CNNConfig, getMNIST_DNNConfig, getBCWConfig, getSpeareConfig, getCIFAR_Res101Config, getCIFAR10_Res34Config
from server.simpleServer import simpleServer as serv

def testAll():

    #MNIST CNN
    print("------------------")
    print("MNIST CNN")
    print("------------------")
    mnistCNNServer = serv(getMNIST_CNNConfig())
    for item in mnistCNNServer.trainTest()[1:]:
        print(item)

    #MNIST DNN
    print("------------------")
    print("MNIST DNN")
    print("------------------")
    mnistDNNServer = serv(getMNIST_DNNConfig())
    for item in mnistDNNServer.trainTest()[1:]:
        print(item)

    #BCW Noiseless
    print("------------------")
    print("BCW Noiseless")
    print("------------------")
    BCWServer = serv(getBCWConfig())
    for item in BCWServer.trainTest()[1:]:
        print(item)

    #BCW w/ Laplace Noise
    print("------------------")
    print("BCW Laplace Noise")
    print("------------------")
    BCWConf = getBCWConfig()
    BCWConf.usePerturbation()
    BCWServer = serv(BCWConf)
    for item in BCWServer.trainTest()[1:]:
        print(item)

    #Speare LSTM
    print("------------------")
    print("Shakespeare LSTM")
    print("------------------")
    speareServer = serv(getSpeareConfig())
    for item in speareServer.trainTest()[1:]:
        print(item)

    
    #CIFAR 100 Resnet 101
    print("------------------")
    print("CIFAR 100 - Resnet 101")
    print("------------------")
    CIFARServer = serv(getCIFAR_Res101Config())
    for item in CIFARServer.trainTest()[1:]:
        print(item)

    #CIFAR 10 Resnet 34
    print("------------------")
    print("CIFAR 10 - Resnet 34")
    print("------------------")
    CIFAR10Server = serv(getCIFAR10_Res34Config())
    for item in CIFAR10Server.trainTest()[1:]:
        print(item)

if(__name__ == '__main__'):
    testAll()
