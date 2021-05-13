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
        kl_anneal = args.kl_anneal
        kl_loss_weight = kl_anneal * args.Batch_Size / x_train.shape[0]

        initializer = tf.keras.initializers.glorot_uniform()

        if args.Arch == 'LSTM':
            inputs = tf.keras.layers.Input(shape=(x_train.shape[1], x_train.shape[2]))
            
            if args.num_layers == 1:
                b_l1 = tf.keras.layers.LSTM(args.sizeof_layers, activation='relu', return_sequences=False,)(inputs)
            if args.num_layers == 2:
                b_l1 = tf.keras.layers.LSTM(args.sizeof_layers, activation='relu', return_sequences=True,)(inputs)
                b_l1 = tf.keras.layers.LSTM(args.sizeof_layers, activation='relu', return_sequences=False,)(b_l1)
            else:
                b_l1 = tf.keras.layers.LSTM(args.sizeof_layers, activation='relu', return_sequences=True,)(inputs)
                for _ in range(args.num_layers-2):
                    b_l1 = tf.keras.layers.LSTM(args.sizeof_layers, activation='relu', return_sequences=True,)(b_l1)
                b_l1 = tf.keras.layers.LSTM(args.sizeof_layers, activation='relu', return_sequences=False,)(b_l1)
                
            base_op = tf.keras.layers.Dense(args.sizeof_bnn, activation = 'relu')(b_l1)
            

        if args.Arch == 'FF':
            inputs = tf.keras.layers.Input(shape=[x_train.shape[1], x_train.shape[2]])
            base_op = tf.keras.layers.Flatten()(inputs)
            for _ in range(args.num_layers):
                base_op = tf.keras.layers.Dense(args.sizeof_layers, activation='relu')(base_op)
                if args.batch_norm:
                    base_op = tf.keras.layers.BatchNormalization()(base_op)
            base_op = tf.keras.layers.Dense(args.sizeof_bnn, activation = 'relu')(base_op)
                

        
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

            if args.batch_norm:
                base_op = tf.keras.layers.BatchNormalization()(base_op)
            ext_l2 = tfp.layers.DenseVariational(units=y_train.shape[1],
                                                 make_posterior_fn=posterior,
                                                 make_prior_fn=prior,
                                                 kl_weight=kl_loss_weight)(base_op)
            ext_op = tfp.layers.DistributionLambda(
                lambda t: tfd.Normal(loc=t,
                                     scale=0.1))(ext_l2)

            loss = NLL
            
        elif args.Ext == '-c':
            def posterior(kernel_size, bias_size=0, dtype=None):
                n = kernel_size + bias_size
                return tf.keras.Sequential([
                    tfp.layers.VariableLayer(2 * n, dtype=dtype),
                    tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                        tfd.Normal(loc=t[..., :n],
                                   scale=1e-5 + softplus(t[..., n:], rho=args.rho_q)),
                        reinterpreted_batch_ndims=1)),
                ])

            def prior(kernel_size, bias_size=0, dtype=None):
                n = kernel_size + bias_size
                return tf.keras.Sequential([
                    tfp.layers.VariableLayer(n, dtype=dtype),
                    tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                        tfd.Normal(loc=t, scale=args.prior_scale),
                        reinterpreted_batch_ndims=1)),
                ])

            if args.batch_norm:
                base_op = tf.keras.layers.BatchNormalization()(base_op)
                
            ext_l2 = tfp.layers.DenseVariational(units=2*y_train.shape[1],
                                                 make_posterior_fn=posterior,
                                                 make_prior_fn=prior,
                                                 kl_weight=kl_loss_weight)(base_op)
            ext_op = tfp.layers.DistributionLambda(
                lambda t: tfd.Normal(loc=t[..., :y_train.shape[1]],
                                     scale=1e-5 + softplus(t[..., y_train.shape[1]:], rho=args.rho_op)))(ext_l2)

            loss = NLL
        
        else:
            return False


        if (args.Ext == '-m') or (args.Ext == '-c'):
            self.model = bayesian_model(inputs=inputs, outputs=ext_op)
            # self.model = tf.keras.Model(inputs=inputs, outputs=ext_op)
        else:
            self.model = tf.keras.Model(inputs=inputs, outputs=ext_op)

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                           loss=loss,
                           )

    def fit(self, x, y, callback=None, verbose=False):
        
        def exp_scheduler(epoch, ):
            if epoch < self.args.scheduler_start_epoch:
                return self.args.lr
            else:
                return self.args.lr * tf.math.exp(self.args.scheduler_exp_val)



        def cos_scheduler(epoch):
            max = 0.001
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
            
        stop_nan = tf.keras.callbacks.TerminateOnNaN()

        self.model.fit(x, y,
                       epochs=self.args.Epochs,
                       batch_size=self.args.Batch_Size,
                       callbacks=[LR_Schedule, stop_nan],
                      verbose=verbose)

        self.history = self.model.history.history

    def predict(self, x, y=None, k=100):
        predictions = pd.DataFrame(index=y.index)
        predictions['True'] = y['ILI']

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

            
        for column in predictions.columns:
            predictions[column] = predictions[column].astype('float')
        return predictions

