import json
import urllib.request
import markets.marketdb

class Poloniex:

    def __init__(self, database):

        self._database = database


    def buildAPIUrl(self, currency, quoteCurrency, fromutc, toutc):
        '''
        Makes a URL for querying historical prices of a cyrpto from Poloniex
        cur:    3 letter abbreviation for cryptocurrency (BTC, LTC, etc)
        '''
        url = 'https://poloniex.com/public?command=returnChartData&currencyPair='+quoteCurrency+'_' + currency + '&start='+str(fromutc)+'&end='+str(toutc)+'&period=300'
        return url

    def loadAndStore(self, currency, quoteCurrency, fromutc, toutc):
        '''
        cur:    3 letter abbreviation for cryptocurrency (BTC, LTC, etc)
        fp:     File path (to save price data to CSV)
        '''
        openUrl = urllib.request.urlopen(self.buildAPIUrl(currency,quoteCurrency,fromutc,toutc))
        r = openUrl.read()
        openUrl.close()
        list = json.loads(r.decode())

        print(list)

        self._database.store(currency,quoteCurrency,"POLONIEX",list)



