import sqlite3
import pandas as pd
import numpy as np
import time

class Database:

    def __init__(self, config):
        self._config = config
        self._databasePath = config["databaseDir"]+"/Data.db"

        self._connection = sqlite3.connect(self._databasePath)

        self._connection.execute('CREATE TABLE IF NOT EXISTS MarketData (utc INTEGER, market varchar(20),'
                                 ' coin varchar(20), quoteCoin varchar(20), high FLOAT, low FLOAT,'
                                 ' open FLOAT, close FLOAT, volume FLOAT, '
                                 ' quoteVolume FLOAT, weightedAverage FLOAT,'
                                 'PRIMARY KEY (utc, coin, quoteCoin, market));')
        self._connection.commit()

        self._quoteCoin = config["quoteCoin"]
        self._coins = config["coins"]
        self._market = "BINANCE"


    def storeDataFrame(self, coin, df):

        for index, row in df.iterrows():

            if index>0:
                self._connection.execute(
                    "INSERT INTO MarketData (utc, market, coin, quoteCoin, high, low, open, close, volume) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (int(index), self._market, coin, self._quoteCoin,
                     row['high'], row['low'], row['open'], row['close'], row['volume']))
        self._connection.commit()


    def rangeUtcstamp(self):

        cursor = self._connection.cursor()
        coins = "('"+self._config["coins"][0]+"'"
        for c in self._config["coins"][1:]:
            coins = coins +",'"+c+"'"
        coins = coins + ")"

        cursor.execute("SELECT MIN(minutc), MIN(maxutc) FROM (SELECT MIN(utc) as minutc,MAX(utc) as maxutc FROM MarketData WHERE quoteCoin = '"+self._config["quoteCoin"]+"' "+
                       "AND coin in "+coins+" GROUP BY coin,quoteCoin)")
        result = cursor.fetchone()

        return result[0],result[1]


    def maxUtcstamp(self, coin):
        cursor = self._connection.cursor()
        cursor.execute("SELECT MAX(utc) FROM MarketData WHERE coin=? and quoteCoin=? and market=?",
                       (coin, self._quoteCoin, self._market))
        result = cursor.fetchone()[0]

        return result

    def test(self):
        cursor = self._connection.cursor()
        sql = "SELECT utc,high,low,close,volume FROM MarketData WHERE utc >= 1519462800 AND coin='ETH' AND quoteCoin='BTC' AND market='BINANCE'"
        cursor.execute(sql)
        result = cursor.fetchone()

        data = pd.read_sql_query(sql, con=self._connection, index_col="utc")
        print(data)

    def readDataFrame(self, coin, fromutc, toutc):

        sql = "SELECT utc,high,low,close,volume FROM MarketData WHERE utc >= "+str(fromutc)+" AND utc <= "+str(toutc)+" AND coin='"+coin+"' AND quoteCoin='"+self._quoteCoin+"' AND market='"+self._market+"'"
        data = pd.read_sql_query(sql, con=self._connection, index_col="utc")
        idx = range(fromutc, toutc+1, self._config["tradeInterval"])
        data = data.reindex(index=idx,fill_value=np.nan)
        data = data.fillna(axis=0, method="bfill")
        data = data.fillna(axis=0, method="ffill")
        data['volume'] = data['volume'].replace(to_replace=0.0,method='bfill') ## replace the few 0 volumes with backfill, cannot deal with zero during normalization further


        return data

    def deleteLast(self, utc):

        self._connection.execute("DELETE FROM MarketData WHERE utc >= "+str(int(utc)))
        self._connection.commit()

    def readAll(self, fromutc, toutc, test=False):


        result = np.zeros(shape=[len(self._coins), int((toutc-fromutc)/self._config["tradeInterval"])+1, 4])

        i=0
        for coin in self._coins:

            coindf = self.readDataFrame(coin, int(fromutc), int(toutc), test)
            result[i,:] = coindf.values
            i+=1

        return result










