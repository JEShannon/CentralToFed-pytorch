from demoConfigs import getMNIST_CNNConfig, getMNIST_DNNConfig, getBCWConfig, getSpeareConfig, getCIFAR_Res101Config, getCIFAR10_Res34Config, getCIFAR10_multiGPU_Res50Config
from server.simpleServer import simpleServer as serv

def main():
    conf = getCIFAR10_multiGPU_Res50Config()
    if(not conf["server"]):
        server = serv(conf)
    else:
        server = conf["server"](conf)
    for item in serv.trainTest()[1:]:
        print(item)


main()
