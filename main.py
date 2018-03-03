import json
from markets import marketdb
from markets import download
from markets import binanceex
from markets import analysis
import logging
import time
import agent
import neural
import webserver

def getConfig():
    with open('./marketsconfig.json') as json_data:
        return json.load(json_data)


logging.basicConfig(level=logging.INFO)

config = getConfig()

db = marketdb.Database(config)

df = db.readDataFrame("NEO",1518213600,int(time.time()))
df = analysis.BollingerBands(df)
df = analysis.CCI(df)
df = analysis.EVM(df,10)

output=[]
webserver = webserver.TraderHTTPServer()
webserver.run(9080,output)

binex = binanceex.Binanceex(config, db)

binex.createPortfolioVector()

downloader = download.Downloader(config,db,binex)
downloader.download()


agent = agent.Agent(config, db, neural.Neural(config), output)

#agent.train()

agent.restore("./model/my_test_model-5000")


print (time.time())

agent.trade()












