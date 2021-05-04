import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
import numpy as np
from bayesian_model import *

tfd = tfp.distributions

class model_builder:
    def __init__(self, x_train, y_train, args):

        self.args=args

        NLL = lambda y, p_y: -p_y.log_prob(y)
        MSE = lambda y, p_y: tf.math.square(y-p_y)
        kl_anneal = 0.1
        kl_loss_weight = kl_anneal * args.Batch_Size / x_train.shape[0]

        initializer = tf.keras.initializers.glorot_uniform()

        if args.Arch == 'LSTM':
            inputs = tf.keras.layers.Input(shape=(x_train.shape[1], x_train.shape[2]))
            b_l1 = tf.keras.layers.LSTM(128, activation='relu', return_sequences=False, kernel_initializer=initializer)(inputs)
            b_l2 = tf.keras.layers.BatchNormalization()(b_l1)
            base_op = tf.keras.layers.Dense(25, activation = 'relu')(b_l2)
        else:
            inputs = tf.keras.layers.Input(shape=[x_train.shape[1], x_train.shape[2]])
            b_l1 = tf.keras.layers.Flatten()(inputs)
            b_l1 = tf.keras.layers.Dense(256, activation='relu')(b_l1)
            base_op = tf.keras.layers.Dense(25, activation='relu')(b_l1)

        if args.Ext == '-v':
            ext_op = tf.keras.layers.Dense(y_train.shape[1])(base_op)
            loss = MSE

        elif args.Ext == '-d':
            ext_l1 = tf.keras.layers.BatchNormalization()(base_op)
            ext_l2 = tf.keras.layers.Dense(2, activation='linear')(ext_l1)
            ext_op = tfp.layers.DistributionLambda(
                lambda t: tfd.Normal(loc=t[..., :y_train.shape[1]],
                                     scale=1e-5 + softplus(t[..., y_train.shape[1]:], rho=0.25)))(ext_l2)

            loss = NLL

        elif args.Ext == '-m':
            def posterior(kernel_size, bias_size=0, dtype=None):
                n = kernel_size + bias_size
                return tf.keras.Sequential([
                    tfp.layers.VariableLayer(2 * n, dtype=dtype),
                    tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                        tfd.Normal(loc=t[..., :n],
                                   scale=1e-5 + softplus(t[..., n:], rho=10.0)),
                        reinterpreted_batch_ndims=1)),
                ])

            def prior(kernel_size, bias_size=0, dtype=None):
                n = kernel_size + bias_size
                return tf.keras.Sequential([
                    tfp.layers.VariableLayer(n, dtype=dtype),
                    tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                        tfd.Normal(loc=t, scale=1.0),
                        reinterpreted_batch_ndims=1)),
                ])

            ext_l1 = tf.keras.layers.BatchNormalization()(base_op)
            ext_l2 = tfp.layers.DenseVariational(units=y_train.shape[1],
                                                 make_posterior_fn=posterior,
                                                 make_prior_fn=prior,
                                                 kl_weight=kl_loss_weight)(ext_l1)
            ext_op = tfp.layers.DistributionLambda(
                lambda t: tfd.Normal(loc=t,
                                     scale=0.5))(ext_l2)

            loss = NLL
        else:
            def posterior(kernel_size, bias_size=0, dtype=None):
                n = kernel_size + bias_size
                return tf.keras.Sequential([
                    tfp.layers.VariableLayer(2 * n, dtype=dtype),
                    tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                        tfd.Normal(loc=t[..., :n],
                                   scale=1e-5 + softplus(t[..., n:], rho=10.0)),
                        reinterpreted_batch_ndims=1)),
                    #
                    # tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                    #     tfd.Uniform(low=-1e-5 - softplus(t[..., :n], rho=1.0),
                    #                high=1e-5 + softplus(t[..., n:], rho=1.0))))
                ])

            def prior(kernel_size, bias_size=0, dtype=None):
                n = kernel_size + bias_size
                return tf.keras.Sequential([
                    tfp.layers.VariableLayer(n, dtype=dtype),
                    tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                        tfd.Normal(loc=t, scale=0.5),
                        reinterpreted_batch_ndims=1)),
                    # tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                    #     tfd.Uniform(low=t*0.01 - 0.5,
                    #     high=t*0.01 + 0.5)))
                ])

            ext_l1 = tf.keras.layers.BatchNormalization()(base_op)
            ext_l2 = tfp.layers.DenseVariational(units=2*y_train.shape[1],
                                                 make_posterior_fn=posterior,
                                                 make_prior_fn=prior,
                                                 kl_weight=kl_loss_weight)(ext_l1)
            ext_op = tfp.layers.DistributionLambda(
                lambda t: tfd.Normal(loc=t[..., :y_train.shape[1]],
                                     scale=1e-5 + softplus(t[..., y_train.shape[1]:], rho=0.25)))(ext_l2)

            loss = NLL


        if (args.Ext == '-m') or (args.Ext == '-c'):
            self.model = bayesian_model(inputs=inputs, outputs=ext_op)
            # self.model = tf.keras.Model(inputs=inputs, outputs=ext_op)
        else:
            self.model = tf.keras.Model(inputs=inputs, outputs=ext_op)

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
                           loss=loss,
                           )

    def fit(self, x, y, callback=None):
        def exp_scheduler(epoch, lr):
            if epoch < 10:
                return lr
            else:
                return lr * tf.math.exp(-0.02)



        def cos_scheduler(epoch):
            max = 0.005
            min = 1e-4
            range = max-min
            warmup = 20
            if epoch < warmup:
                return min + epoch*range/warmup
            else:
                return range/2 * tf.math.cos(3.1415*(epoch-warmup)/(200-warmup)) + range/2 + min

        if self.args.Arch == 'FF':
            LR_Schedule = tf.keras.callbacks.LearningRateScheduler(exp_scheduler)
        else:
            LR_Schedule = tf.keras.callbacks.LearningRateScheduler(cos_scheduler)

        self.model.fit(x, y,
                       epochs=self.args.Epochs,
                       batch_size=self.args.Batch_Size,
                       callbacks=[LR_Schedule])

        self.history = self.model.history.history

    def predict(self, x, y=None, k=100):
        predictions = pd.DataFrame(index=y.index)
        predictions['True'] = y['T0']

        if self.args.Ext == '-v':
            predictions['Pred'] = self.model(x).numpy()

        elif self.args.Ext == '-d':
            yhat = self.model(x)
            predictions['Pred'] = yhat.mean().numpy()
            predictions['Std'] = yhat.stddev().numpy()

        elif self.args.Ext == '-m':
            yhats = [self.model(x) for _ in range(k)]

            means = []
            for yhat in yhats:
                means.append(np.squeeze(yhat.mean().numpy()))

            predictions['Pred'] = np.mean(means, 0)
            predictions['Std'] = np.std(means, 0)

        else:
            yhats = [self.model(x) for _ in range(k)]

            means = []
            var = []
            for yhat in yhats:
                means.append(np.squeeze(yhat.mean().numpy()))
                var.append(np.squeeze(yhat.variance().numpy()))

            predictions['Pred'] = np.mean(means, 0)
            predictions['Std'] = np.sqrt(np.mean(np.square(means), 0) - np.square(np.mean(means, 0)) + np.mean(var,0))

        return predictions

