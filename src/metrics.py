import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import pearsonr
from scipy import special
import pandas as pd
import numpy as np
tfd = tfp.distributions


def _normpdf(x):
    """Probability density function of a univariate standard Gaussian
    distribution with zero mean and unit variance.
    """
    _normconst = 1.0 / np.sqrt(2.0 * np.pi)
    return _normconst * np.exp(-(x * x) / 2.0)

def crps(true, mean=None, std=None, grad=False):
    """
    Computes the CRPS of observations x relative to normally distributed
    forecasts with mean, mu, and standard deviation, sig.
    CRPS(N(mu, sig^2); x)
    Formula taken from Equation (5):
    Calibrated Probablistic Forecasting Using Ensemble Model Output
    Statistics and Minimum CRPS Estimation. Gneiting, Raftery,
    Westveld, Goldman. Monthly Weather Review 2004
    http://journals.ametsoc.org/doi/pdf/10.1175/MWR2904.1
    Parameters
    ----------
    x : scalar or np.ndarray
        The observation or set of observations.
    mu : scalar or np.ndarray
        The mean of the forecast normal distribution
    sig : scalar or np.ndarray
        The standard deviation of the forecast distribution
    grad : boolean
        If True the gradient of the CRPS w.r.t. mu and sig
        is returned along with the CRPS.
    Returns
    -------
    crps : scalar or np.ndarray or tuple of
        The CRPS of each observation x relative to mu and sig.
        The shape of the output array is determined by numpy
        broadcasting rules.
    crps_grad : np.ndarray (optional)
        If grad=True the gradient of the crps is returned as
        a numpy array [grad_wrt_mu, grad_wrt_sig].  The
        same broadcasting rules apply.
    """
    try:
        if isinstance(true, pd.DataFrame):
            mean = true['Pred']
            std = true['Std']
            true = true['True']
        _normcdf = special.ndtr

        true = np.asarray(true)
        mean = np.asarray(mean)
        std = np.asarray(std)
        # standadized x
        sx = (true - mean) / std
        # some precomputations to speed up the gradient
        pdf = _normpdf(sx)
        cdf = _normcdf(sx)
        pi_inv = 1. / np.sqrt(np.pi)
        # the actual crps
        crps = std * (sx * (2 * cdf - 1) + 2 * pdf - pi_inv)
        if grad:
            dmu = 1 - 2 * cdf
            dsig = 2 * pdf - pi_inv
            return crps, np.array([dmu, dsig])
        else:
            return crps.mean()

    except:
        return np.nan

def nll(true, mean=None, std=None):
    try:
        if isinstance(true, pd.DataFrame):
            mean = true['Pred']
            std = true['Std']
            true = true['True']
        p_y = tfp.distributions.Normal(mean,std)
        return (-p_y.log_prob(true)).numpy().mean()
    except:
        return np.nan


def mae(true, pred=None):
    if isinstance(true, pd.DataFrame):
        pred = true['Pred']
        true = true['True']

    return tf.reduce_mean(tf.math.abs(true-pred)).numpy()

def rmse(true, pred=None):
    if isinstance(true, pd.DataFrame):
        pred = true['Pred']
        true = true['True']

    return tf.sqrt(tf.reduce_mean(tf.square(true-pred))).numpy()

def smape(true, pred=None):
    if isinstance(true, pd.DataFrame):
        pred = true['Pred'].values
        true = true['True'].values

    return 100 / len(true) * np.sum(np.abs(pred - true) / (np.abs(true) + np.abs(pred)))

def corr(true, pred=None):
    if isinstance(true, pd.DataFrame):
        pred = true['Pred'].values
        true = true['True'].values

    if type(pred) != np.ndarray:
        pred = true.numpy()

    pred = pred.astype('float32')
    true = true.astype('float32')
    corr = pearsonr(true, pred)[0]
    return corr

def mb_log(true, mean = None, std = None):
    try:
        if isinstance(true, pd.DataFrame):
            mean = true['Pred']
            std = true['Std']
            true = true['True']

        dist = tfd.Normal(loc=mean, scale=std)

        bins = np.linspace(-0.5, 0.5, 11) / 100
        t_error = []
        for bin in bins:
            t_error.append(dist.prob(true + true * bin).numpy())
        t_error = np.asarray(t_error)


        return np.log(t_error.sum(0)).mean()
    except:
        return np.nan


def smooth(y):
    y_prime = []
    for i in range(7, y.shape[0] - 7):
        y_prime.append(np.mean(y[i - 7:i + 8]))
    return np.asarray(y_prime)

def sdp(true, pred=None):
    if isinstance(true, pd.DataFrame):
        pred = true['Pred'].values
        true = true['True'].values

    y_true_prime = smooth(true)
    y_pred_prime = smooth(pred)

    return np.argmax(y_true_prime) - np.argmax(y_pred_prime)
