import json
from markets import marketdb
from markets import download
from markets import binanceex
from markets import analysis
import logging
import agent
import neural
import webserver

def getConfig():
    with open('./marketsconfig.json') as json_data:
        return json.load(json_data)


logging.basicConfig(level=logging.INFO)

config = getConfig()

db = marketdb.Database(config)

output=[]
webserver = webserver.TraderHTTPServer()
webserver.run(9080,output) ## start a simple webserver on localhost:9080 and follow the online progress while trading every hour

binex = binanceex.Binanceex(config, db)

binex.createPortfolioVector()

downloader = download.Downloader(config,db,binex)
downloader.download()

agent = agent.Agent(config, db, neural.Neural(config), output)

agent.train()

agent.restore("./model/my_test_model-5000")

agent.trade()












