import time as tm
import numpy as np
import tensorflow as tf
import markets.download as download
import replay


class Agent:

    def __init__(self, config, db, neural, weboutput=[]):

        self._config = config
        self._database = db
        self._neural = neural
        self._weboutput = weboutput

        self._batchSize = int(self._config["batchSize"])
        self._windowSize = int(self._config["windowSize"])
        self._numCoins = len(self._config["coins"])
        self._learningRate = self._config["learningRate"]
        self._decayRate = self._config["decayRate"]
        self._decaySteps = self._config["decaySteps"]
        self._trainTestSplit = self._config["trainTestSplit"]

        self._interval = self._config["tradeInterval"]
        self._startutc, self._endutc = self._database.rangeUtcstamp()
        self._startutc = max(self._config["startUtc"], self._startutc)

        self._allX = self._database.readAll(self._startutc, self._endutc, False)

        self._replayMemory = replay.ReplayMemory(config,fromutc=self._startutc,toutc=self._endutc)

        self._commission = self._config["commission"]

        self._startTrainUtc = self._startutc
        self._endTrainUtc =  self._startutc + int(self._trainTestSplit * (self._endutc - self._startutc))
        self._endTrainUtc =  self._endTrainUtc - (self._endTrainUtc % self._interval)

        self._startTestUtc = self._endTrainUtc
        self._endTestUtc = self._endutc

        self.initTensors()

        self._saver = tf.train.Saver(max_to_keep=5)


    def initTensors(self):

        self._neural.buildModel()

        self._numSamples = tf.shape(self._neural.X)[0]

        self._y = tf.placeholder(tf.float32, shape=[None, self._numCoins]);

        self._futurePrice = tf.concat([tf.ones([self._numSamples, 1]), self._y[:, :]], 1)

        self._futureW = (self._futurePrice * self._neural.softmaxW) / tf.reduce_sum(self._futurePrice * self._neural.softmaxW, axis=1)[:, None]

        self._mu = self.transactionRemainder(self._futureW, self._neural.softmaxW)

        self._pv_vector = tf.reduce_sum(self._neural.softmaxW * self._futurePrice, axis=1) * (tf.concat([tf.ones(1), self._mu], axis=0))

        self._portfolio_value = tf.reduce_prod(self._pv_vector)

        self._loss = -tf.reduce_mean(tf.log(self._pv_vector))

        self._globalStep = tf.Variable(0, trainable=False)




    def train(self):

        learningRate = tf.train.exponential_decay(self._learningRate, self._globalStep, self._decaySteps,
                                                        self._decayRate,
                                                        staircase=True)

        trainStep = tf.train.AdamOptimizer(learningRate).minimize(self._loss, global_step=self._globalStep)

        init = tf.global_variables_initializer()


        tf.summary.scalar("cash_bias", self._neural.cashBias[0,0])

        summary_op = tf.summary.merge_all()

        writer = tf.summary.FileWriter("./logs", graph=tf.get_default_graph())

        # Start training
        with tf.Session() as sess:

            self._sess = sess

            # Run the initializer
            sess.run(init)

            ##testBatch = self.buildBatch(self._allX, self.utc2idx(self._startTestUtc), int((self._endTestUtc - self._startTestUtc) / self._interval) - self._windowSize - 1 )

            for step in range(1, 5001):

                startIdx = self.sampleStartIdx(self._startTrainUtc, self._endTrainUtc)

                batch = self.buildBatch(self._allX, startIdx, self._batchSize)

                # Run optimization op (backprop)

                res = sess.run([self._neural.softmaxW, self._loss, trainStep, self._globalStep, summary_op],
                               feed_dict={  self._neural._X: batch["X"],
                                            self._neural.prevW: batch["prevW"],
                                            self._neural.training: True,
                                            self._y: batch["y"]
                                            })

                self.storeW(startIdx, res[0])

                writer.add_summary(res[-1], res[-2])

                ##print("step " + str(res[-1])+" "+str(res[1])) ##+" "+str(res[4]))



                if step % 10 == 0:
                    ## need to rebuild testbatch because of prevW
                    testBatch = self.buildBatch(self._allX, self.utc2idx(self._startTestUtc), int((self._endTestUtc - self._startTestUtc) / self._interval) - self._windowSize - 1)
                    ##testBatch = batch

                    evalW,evalLoss,pval = sess.run([self._neural.softmaxW, self._loss,self._portfolio_value],
                                   feed_dict={self._neural._X: testBatch["X"],
                                              self._neural.prevW: testBatch["prevW"],
                                              self._neural.training: False,
                                              self._y: testBatch["y"]})

                    self.storeW(self.utc2idx(self._startTestUtc), evalW)

                    #print(evalW[-1])

                    #print("step "+str(step)+" loss "+str(evalLoss)+" pval "+str(pval))

                if step % 1000 == 0:
                    ## need to rebuild testbatch because of prevW

                    for k in range(0,5):
                        prevOmaga = np.zeros(shape=[self._numCoins+1])
                        prevOmaga[0] = 1
                        testport = 1
                        maxP = 0
                        maxIdx =0
                        rand = int(np.random.uniform(0,100))

                        for i in range(0,300):

                            testBatch = self.buildBatch(self._allX, self.utc2idx(self._startTestUtc)+i+rand, 1)

                            evalW, = sess.run([self._neural.softmaxW],
                                                         feed_dict={self._neural._X: testBatch["X"],
                                                                    self._neural.prevW: prevOmaga[np.newaxis,1:],
                                                                    self._neural.training: False})

                            if evalW[0][0] > 0.01:

                                self.printW(evalW[0])

                            future_price = np.concatenate((np.ones(1), testBatch["y"][-1,:]))

                            pv_after_commission = self.calculateTransactionRemainder(evalW[0], prevOmaga, self._commission)

                            portfolio_change = pv_after_commission * np.dot(evalW[0], future_price)

                            testport *= portfolio_change

                            if testport > maxP:
                                maxP = testport
                                maxIdx = i

                            ##print("\t"+str(int(testport*100+0.5)-100))

                            prevOmaga = pv_after_commission * evalW[0] * future_price / portfolio_change


                        print("Test step " + str(step) + " loss " + str(evalLoss) + " testport " + str(testport)+" max "+str(maxP)+" idx "+str(maxIdx))

                    self._saver.save(sess, 'model/my_test_model', global_step=self._globalStep)

            print("Optimization Finished!")

            self._replayMemory.save()


    def transactionRemainder(self, futureW, outputW):

        c = self._commission
        w_t = futureW[:self._numSamples-1]  # rebalanced
        w_t1 = outputW[1:self._numSamples]
        mu = 1 - tf.reduce_sum(tf.abs(w_t1[:, 1:]-w_t[:, 1:]), axis=1)*c

        return mu


    def buildBatch(self, allX, startIdx, size):

        ## allX (coin,time,feature(min,max,close,vol) )

        M = np.zeros(shape=[size, self._numCoins, self._windowSize + 1, 3]) ## need +1 because Y needs to be sampled one (1) timestep after the input X (which needs "windowSize" timesteps)

        for i in range(0, size):

            M[i, :, :, :] = allX[:, startIdx + i:startIdx + i + self._windowSize + 1, 0:3]

        X = M[:,:,:-1,:]  ## strip last time col
        X = X[:,:,:,0:3]/X[:,:,-1,2,None,None]

        y = M[:, :, -1, 2] / M[:, :, -2, 2]  ## last close price / last-1 close price

        w_prev = self._replayMemory.getExperiences(np.arange(startIdx, startIdx+size)-1)

        return {'X': X, 'y': y, 'prevW': w_prev}

    def buildInput(self, allX, startIdx, size):

        M = np.zeros(shape=[size, self._numCoins, self._windowSize, 3])

        for i in range(0, size):
            M[i, :, :, :] = allX[:, startIdx + i:startIdx + i + self._windowSize, 0:3]

        print("V(t): ")
        print(M[:, :, -1, 2])
        print("V(t-1):")
        print(M[:, :, -2, 2])

        X = M[:,:,:,0:3]/M[:,:,-1,2,None,None]
        y = M[:, :, -1, 2] / M[:, :, -2, 2]

        print("y: ")
        print(y)

        return {'X': X, 'y': y}

    def storeW(self, idx, w):

        self._replayMemory.addExperiences(fromIdx=idx, w=w[:,1:])

    def sampleStartIdx(self, start, end):

        return int(np.random.uniform(1, (self._endTrainUtc - self._startTrainUtc) / self._interval - self._batchSize - self._windowSize - 1))

    def utc2idx(self, utc):

        return int((utc - self._startutc)/self._interval)

    def calculateTransactionRemainder(self, w1, w0, commission_rate):
        """
        @:param w1: target portfolio vector, first element is btc
        @:param w0: rebalanced last period portfolio vector, first element is btc
        @:param commission_rate: rate of commission fee, proportional to the transaction cost
        """
        mu0 = 1
        mu1 = 1 - 2 * commission_rate + commission_rate ** 2
        while abs(mu1 - mu0) > 1e-10:
            mu0 = mu1
            mu1 = (1 - commission_rate * w0[0] -
                   (2 * commission_rate - commission_rate ** 2) *
                   np.sum(np.maximum(w0[1:] - mu1 * w1[1:], 0))) / \
                  (1 - commission_rate * w1[0])
        return mu1

    def restore(self, model=None):

        #saver = tf.train.import_meta_graph(model+'.meta')
        #saver = tf.train.Saver()

        self._sess = tf.Session()

        if model:
            self._saver.restore(self._sess, model )
        else:
            self._saver.restore(self._sess, tf.train.latest_checkpoint('./model'))

        self._replayMemory.restore()

    def trade(self):

        prevW = np.zeros(shape=[self._numCoins + 1])
        prevW[0] = 1
        evalW = prevW
        mu = 1

        testport = 1

        downloader = download.Downloader(self._config, self._database)

        while True:

            now = int(tm.time())

            wait = self._interval - (now % self._interval)

            closeTime = now + wait

            print("Sleeping " + str(wait) + " secs until close time "+str(closeTime))
            tm.sleep(wait)

            downloadTime = downloader.download(closetime=closeTime-1)

            print("download utc "+str(downloadTime)+" compare to close time - interval "+str(closeTime-self._interval))

            while downloadTime < closeTime - self._interval:  ## downloadtime is start of interval
                print("Data not ready, sleeping 5 more secs")
                tm.sleep(5)
                downloadTime = downloader.download()

            print("success")

            if self._endutc + self._interval <= closeTime:

                Xnew = self._database.readAll(self._endutc + self._interval, closeTime - self._interval)
                self._allX = np.concatenate([self._allX, Xnew], axis=1)
                self._endutc = closeTime - self._interval

                startInputWindow = closeTime - (self._windowSize - 0) * self._interval

                input = self.buildInput(self._allX, self.utc2idx(startInputWindow), 1)

                print("---input---")
                print(str(startInputWindow)+" idx "+str(self.utc2idx(startInputWindow))+" endUtc "+str(self._endutc)+" startUtc "+str(self._startutc))
                print(input)

                price = np.concatenate((np.ones(1), input["y"][-1, :]))

                mu = self.calculateTransactionRemainder(evalW, prevW, self._commission)

                portfolio_change = mu * np.dot(evalW, price)

                testport *= portfolio_change

                print("test port "+str(testport)+" mu "+str(mu)+" change "+str(portfolio_change))

                self._weboutput.append("BTC portfolio value "+str(testport)+"</br></br>")

                prevW = mu * evalW * price / portfolio_change  ## w'(t)

                netoutput, = self._sess.run([self._neural.softmaxW],
                                  feed_dict={self._neural._X: input["X"],
                                             self._neural.prevW: prevW[np.newaxis,1:],
                                             self._neural.training: False})

                evalW = netoutput[0]  ## w(t)

                self.printW(evalW)



                #prevW = evalW[0]





    def printW(self, w):

        coins = self._config["coins"][:]
        coins.insert(0,self._config["quoteCoin"])


        line=""
        i=0
        for v in w:
            if v > 0.0000:
                line = line + coins[i]+" "+str(int(v*100+0.5))+" "
            i=i+1

        self._weboutput.append(tm.strftime("%Y-%m-%d %H:%M:%S", tm.localtime())+' '+line+'</br>')

        print(line)







