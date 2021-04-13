import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
tfd = tfp.distributions

loss_tracker = tf.keras.metrics.Mean(name='loss')
kl_tracker = tf.keras.metrics.Mean(name='kl')
likelihood_tracker = tf.keras.metrics.Mean(name='likelihood')

@tf.function
def softplus(x, rho=1.0):
    # softplus function with sharpening factor rho
    # ln(1+exp(kx))/rho
    return tf.math.divide(tf.math.log(tf.math.add(1.0, tf.math.exp(tf.math.multiply(x, rho)))), rho)

NLL = lambda y, p_y: -p_y.log_prob(y)

class bayesian_model(tf.keras.Model):
    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self.layers[0](x, training=True)
            kl = 0
            for layer in self.layers[1:]:
                y_pred = layer(y_pred, training=True)

                if layer.name == 'dense_variational':
                    dtype = tf.as_dtype(layer.dtype or tf.keras.backend.floatx())
                    inputs = tf.cast(y_pred, dtype, name='inputs')
                    q = layer._posterior(inputs)
                    r = layer._prior(inputs)
                    kl = kl + layer._kl_divergence_fn(q, r)

            likelihood = NLL(y, y_pred)
            loss = likelihood + kl

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        loss_tracker.update_state(loss)
        kl_tracker.update_state(kl)
        likelihood_tracker.update_state(likelihood)
        return {'Loss': loss_tracker.result(),
                'Likelihood': likelihood_tracker.result(),
                'KL': kl_tracker.result()}

    @property
    def metrics(self):
        return [loss_tracker, likelihood_tracker, kl_tracker]


class bayesian_model_int(tf.keras.Model):
    def train_step(self, data):
        x, y = data
        k=10
        with tf.GradientTape() as tape:
            y_pred = self.layers[0](x, training=True)
            kl = 0
            flag = False
            for layer in self.layers[1:]:
                if layer.name == 'dense_variational':
                    dtype = tf.as_dtype(layer.dtype or tf.keras.backend.floatx())
                    inputs = tf.cast(y_pred, dtype, name='inputs')
                    q = layer._posterior(inputs)
                    r = layer._prior(inputs)
                    kl = kl + layer._kl_divergence_fn(q, r)
                    yhats = [layer(y_pred, training=True) for _ in range(k)]
                    flag = True


                else:
                    y_pred = layer(y_pred, training=True)

                    if flag:
                        yhats = [layer(pred, training=True) for pred in yhats]
                        means = []
                        var = []
                        for yhat in yhats:
                            means.append(tf.squeeze(yhat.mean()))
                            var.append(tf.squeeze(yhat.variance()))

                        y_pred = tfd.Normal(loc = tf.reduce_mean(means, 0),
                                            scale = tf.sqrt(tf.math.reduce_mean(tf.square(means), 0)
                                                            - tf.square(tf.reduce_mean(means, 0))
                                                            + tf.reduce_mean(var, 0)))

            likelihood = NLL(y, y_pred)
            loss = likelihood + kl

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        loss_tracker.update_state(loss)
        kl_tracker.update_state(kl)
        likelihood_tracker.update_state(likelihood)
        return {'Loss': loss_tracker.result(),
                'Likelihood': likelihood_tracker.result(),
                'KL': kl_tracker.result()}

    @property
    def metrics(self):
        return [loss_tracker, likelihood_tracker, kl_tracker]

