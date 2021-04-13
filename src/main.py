import tensorflow as tf
import numpy as np
import metrics
import time
from parser import *
from data_builder import *
from model import *

import matplotlib.pyplot as plt
parser = parser()
args = parser.parse_args()



results = pd.DataFrame(index = ['CRPS','NLL','MAE','RMSE','SMAPE','Corr','MB Log','SDP'])
for fold_num in range(1,2):
    data = data_builder(args, fold=fold_num)
    x_train, y_train, x_test, y_test = data.build()

    days_test = 365
    days_train = 365
    x_test = x_test[-days_test:,:, -30:]
    # y_test = y_test[-days_test:,:]

    x_train = x_train[-days_train:,:, -30:]
    y_train = y_train[-days_train:,:]

    model = model_builder(x_train, y_train, args)

    model.fit(x_train, y_train)
    pred = model.predict(x_test, y_test)

    results[str(fold_num + 2013) + '/' + str(fold_num + 14)] = [metrics.crps(pred),
                                                                metrics.nll(pred),
                                                                metrics.mae(pred),
                                                                metrics.rmse(pred),
                                                                metrics.smape(pred),
                                                                metrics.corr(pred),
                                                                metrics.mb_log(pred),
                                                                metrics.sdp(pred)]
    tf.keras.backend.clear_session()

results['Average'] = results.mean(1)
results['Average'].loc['SDP'] = np.abs(results.loc['SDP'].values[-1]).mean()

plt.plot(pred.index, pred['True'], color='black')
plt.plot(pred.index, pred['Pred'], color = 'red')
plt.fill_between(pred.index, pred['Pred']-pred['Std'], pred['Pred']+pred['Std'], color='pink', alpha=0.5)
plt.show()
