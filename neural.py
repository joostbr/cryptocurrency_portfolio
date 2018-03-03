import tensorflow as tf
import numpy as np

class Neural:

    def __init__(self, config):

        self._config = config

        self._windowSize = int(self._config["windowSize"])
        self._numCoins = len(self._config["coins"])


    def buildModel(self):

        numFilters1 = 6
        filterSize1 = 3
        numFilters2 = 10
        filterSize2 = self._windowSize - (filterSize1-1)

        self._training = tf.placeholder(tf.bool)

        self._X = tf.placeholder("float", [None, self._numCoins, self._windowSize, 3], name="X")
        input_dims = tf.shape(self._X)

        self._prevW = tf.placeholder(tf.float32, shape=[None, self._numCoins])

        conv1 = self.conv_layer(self._X, [1, filterSize1, 3, numFilters1], self._training, name="conv_1")

        conv2 = self.conv_layer(conv1, [1, filterSize2, numFilters1, numFilters2], self._training, name="conv_2")

        w_prev_reshaped = tf.reshape(self._prevW, [-1, self._numCoins, 1, 1])

        w_concat = tf.concat([conv2, w_prev_reshaped], axis=3)  ## concat w in de depth direction (axis 3), cfr fig 2 paper

        conv3 = self.conv_layer(w_concat, [1,1,numFilters2+1,1], self._training, name="conv_3")  ## numFilters2 + 1 since we appended w_prev

        flatten = conv3[:, :, 0, 0]

        self._cashBias = tf.get_variable("cash_bias", [1, 1], dtype=tf.float32, initializer=tf.zeros_initializer)

        tiled_cash_bias = tf.tile(self._cashBias, [input_dims[0], 1])

        voting = tf.concat([tiled_cash_bias, flatten], 1)

        ##full1 = self.layer(voting, [self._numCoins+1, 64], phase_train=self._training, name="fully_1")

        ##full2 = self.layer(full1, [64, self._numCoins + 1], phase_train=self._training, name="fully_2")

        self._softmax = tf.nn.softmax(voting)



    def conv_layer(self, x, weight_shape, training, name="convolution"):  ##filter_size(B x H x D) x num_filters

        with tf.variable_scope(name):
            insize = np.prod(weight_shape)

            W = tf.get_variable("W", weight_shape, initializer=tf.random_normal_initializer(stddev=(2.0 / insize) ** 0.5))
            b = tf.get_variable("b", weight_shape[-1], initializer=tf.constant_initializer(value=0))

            conv = tf.nn.conv2d(x, W, [1, 1, 1, 1], padding='VALID')

            logits = tf.nn.bias_add(conv,b)

            act = tf.nn.relu( self.conv_batch_norm( logits, weight_shape[3], training) )

            ##act = tf.nn.relu( logits )

        return act


    def conv_batch_norm(self, x, n_out, phase_train):
        beta_init = tf.constant_initializer(value=0.0,
                                            dtype=tf.float32)
        gamma_init = tf.constant_initializer(value=1.0,
                                             dtype=tf.float32)
        beta = tf.get_variable("beta", [n_out],
                               initializer=beta_init)
        gamma = tf.get_variable("gamma", [n_out],
                                initializer=gamma_init)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2],
                                              name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.9)
        ema_apply_op = ema.apply([batch_mean, batch_var])
        ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)

        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)


        mean, var = tf.cond(phase_train, mean_var_with_update, lambda: (ema_mean, ema_var))

        normed = tf.nn.batch_norm_with_global_normalization(x, mean, var, beta, gamma, 1e-3, True )

        return normed

    def layer(self, input, weight_shape, phase_train, name):

        with tf.variable_scope(name):

            weight_init = tf.random_normal_initializer(stddev=(1.0 / weight_shape[0]) ** 0.5)
            bias_init = tf.constant_initializer(value=0)

            W = tf.get_variable("W", weight_shape, initializer=weight_init)
            b = tf.get_variable("b", weight_shape[1], initializer=bias_init)  ## always use one bias per hidden neuron

            logits = tf.matmul(input, W) + b

            return tf.nn.relu(logits)

    @property
    def softmaxW(self):
        return self._softmax

    @property
    def prevW(self):
        return self._prevW


    @property
    def X(self):
        return self._X

    @property
    def training(self):
        return self._training

    @property
    def cashBias(self):
        return self._cashBias




