from binance.client import Client
import pandas
import numpy as np
import time
import utils
import logging

class Binanceex:

    def __init__(self, config, database):

        self._client = Client("keyA", "keyB")
        self._database = database
        self._config = config
        self._quoteCoin = config["quoteCoin"]

    def createDataFrame(self, klines, reverse=False):

        y = np.array([np.array(xi[0:8]) for xi in klines])
        idx = (y[:, 0].astype(int) / 1000).astype(int)

        y = y.astype(np.float64)
        if reverse:
            for row in y:
                row[5] = row[7]
                row[1] = 1.0 / row[1]
                row[2] = 1.0 / row[2]
                row[3] = 1.0 / row[3]
                row[4] = 1.0 / row[4]

        df = pandas.DataFrame(data = y[:,1:6], index = idx, columns=['open','high','low','close','volume'])
        df['open'] = pandas.to_numeric(df['open'])
        df['close'] = pandas.to_numeric(df['close'])
        df['high'] = pandas.to_numeric(df['high'])
        df['low'] = pandas.to_numeric(df['low'])
        df['volume'] = pandas.to_numeric(df['volume'])

        return df

    def loadAndStore(self, currency, fromutc=None, toutc=None):

        reverse = False
        if currency == "USDT" and self._quoteCoin == "BTC":
            reverse = True

        if fromutc == None:
            last = self._database.maxUtcstamp(currency)
            if last==None:
                last = utils.parse_time(self._config["startDate"])
            if reverse:
                ticker = self._quoteCoin+""+currency
            else:
                ticker = currency+""+self._quoteCoin
        else:
            last = fromutc

        if toutc == None:
            toutc = time.time()

        logging.info("Downloading "+currency+" from "+utils.format_time(last))

        interval = self._config["tradeInterval"]
        if interval == 3600:
            klines = self._client.get_historical_klines2(ticker, Client.KLINE_INTERVAL_1HOUR, (last+1)*1000, int(toutc)*1000)
        elif interval == 1800:
            klines = self._client.get_historical_klines2(ticker, Client.KLINE_INTERVAL_30MINUTE, (last + 1) * 1000, int(toutc) * 1000)
        else:
            print("unsupported interval "+interval)
            exit(-1)

        if len(klines) > 0:
            df = self.createDataFrame(klines, reverse)
            print(df)
            self._database.storeDataFrame(currency, df)

            return max(df.index)

        else:

            return last

    def getBalance(self, prices=None):

        if not prices:
            prices = self.getActualPrices()

        info = self._client.get_account()
        result = {}
        coins = list(self._config["coins"])
        coins.append(self._quoteCoin)
        for coin in coins:
            for bal in info["balances"]:
                if bal["asset"] == coin:
                    qty = float(bal["free"])+float(bal["locked"])
                    amount = qty * prices[coin]
                    result[coin] = {"qty": qty, "amount": amount}


        print(result)

        return result

    def getActualPrices(self):
        result = {}
        tickers = self.getAllTickers()
        for coin in self._config["coins"]:
            if coin == "USDT":
                ticker = self._quoteCoin+"USDT"
            else:
                ticker = coin + self._quoteCoin

            for each in tickers:
                if each["symbol"]==ticker:
                    if coin == "USDT":
                        result[coin]=1.0/float(each["price"])
                    else:
                        result[coin] = float(each["price"])

        result[self._quoteCoin]=1.0

        return result


    def getAllTickers(self):

        result = self._client.get_all_tickers()

        return result

    def createPortfolioVector(self):

        sum = 0
        balance = self.getBalance()
        for key in balance:
            sum = sum + balance[key]["amount"]

        result = np.zeros(shape=[len(self._config["coins"])+1])
        result[0] = balance[self._quoteCoin]["amount"]/sum

        i=1
        for coin in self._config["coins"]:
            result[i] = balance[coin]["amount"]/sum
            i+=1

        print(result)
        return result














