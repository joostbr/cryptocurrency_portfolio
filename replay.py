import pandas as pd
import numpy as np

class ReplayMemory:

    def __init__(self, config, fromutc, toutc):

        self._config = config
        self._numCoins = len(config["coins"])
        self._tradeInterval = config["tradeInterval"]
        self._fromutc = fromutc

        mem = np.ones(shape=[int((toutc-fromutc)/self._tradeInterval),self._numCoins])*1.0/self._numCoins
        index = np.arange(0,int((toutc-fromutc)/self._tradeInterval))

        self._memory = pd.DataFrame(index=index, columns=[config["coins"]], data=mem)


    def addExperiences(self, fromIdx, w):

        for idx in range(fromIdx, fromIdx+len(w)):
            self._memory.loc[idx] = w[idx-fromIdx]


    def addExperience(self, idx, w):
        self._memory.loc[idx] = w

    def getExperiences(self, indexes):
        return self._memory.loc[indexes].values

    def save(self):

        self._memory.to_csv("./model/portmemory.csv")

    def restore(self):

        self._memory = pd.DataFrame.from_csv("./model/portmemory.csv")




