from demoConfigs import getMNIST_CNNConfig, getMNIST_DNNConfig, getBCWConfig, getSpeareConfig, getCIFAR_Res101Config, getCIFAR10_Res34Config
from server.simpleServer import simpleServer as serv

def main():
    mnistCNNServer = serv(getMNIST_CNNConfig())
    for item in mnistCNNServer.trainTest()[1:]:
        print(item)


main()
