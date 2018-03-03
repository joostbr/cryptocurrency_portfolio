from markets.binanceex import Binanceex
import time as tm

class Downloader:

    def __init__(self, config, db, binancex):
        self._db = db
        self._config = config
        self._binex = binancex
        #self._binex = Binanceex(config, db)

    def download(self, closetime=None):

        minTime = None

        if closetime == None:
            now = tm.time()
            closetime = int(now - (now % self._config["tradeInterval"]))

        while True:

            try:

                for coin in self._config["coins"]:
                    time = self._binex.loadAndStore(coin, fromutc=None, toutc=closetime - 1)
                    if minTime == None or time < minTime :
                        minTime = time

                if minTime == None:
                    return 0
                else:
                    return minTime

            except:

                print("HTTP error downloading crypto data")



