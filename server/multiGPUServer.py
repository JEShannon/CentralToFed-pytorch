import torch.multiprocessing as mp
from torch.multiprocessing import Process, Pipe
import sys

from .serverBase import serverBase

"""
NOTE:  From my capstone I learned that pipes are terrible for moving
  models around due to their max data transfer rates.  Instead the
  models will be saved to a low latency SSD/HDD cache, with safeguards
  to prevent errors.  If there is a better way to do this let me know
  in a github issue.
"""

class multiGPUServer(serverBase):

    def __init__(self, config):
        if(not config["gpuCount"] or config["gpuCount"] < 1):
            #for now this isn't an error just fixing it, but consider changing that.
            print("Invalid GPU count detected, make sure to adjust the configuration you're using!",file=sys.stderr)
            print("WARN: GPU count set to 1")
            count = 1
        else:
            count = config["gpuCount"]
        if(not config["gpuIds"]):
            print("WARN: No GPU Ids given, will default to cuda device(s) [", "0" if (config["gpuCount"] == 1) else ("0-"+str(count)), "]", sep = "", file=sys.stderr)
            ids = "0" if (config["gpuCount"] == 1) else ("0-"+str(count))
        else:
            ids = config["gpuIds"]
        if isinstance(ids, str):
            name = "GPUServer_"+count+"gpus_"+ids
        else:
            name = "GPUServer_"+count+"gpus"
            for ID in ids:
                name += "_"+ID
        super().__init__(name)
        self.__config = config
        numClients = 10   #### TO DO:  Add a value to track this value instead
        self.__model = config["model"]
        data, testset = config["dataFn"](numClients)
        self.__datasets = data
        test_set = [next(iter(DataLoader(testset, batch_size=50, shuffle=True)))]
        self.__testSet = test_set
        self.__aggregator = config["aggregator"]
        self.__clients = []
        #first ready the threads - one for each GPU.  We assume that the gpuCount and gpuIds (if provided) are accurate
        for i in range(numClients):
            #initialize the clients
            #### TODO: Use the config's clients array to have different clients in the same system -- NEEDED for scenarios with malicious clients.
            #### TODO: Initialize the clients on each GPU.  Each GPU gets one thread to feed it, and dedicated users for each thread
