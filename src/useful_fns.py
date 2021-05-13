import datetime as dt
import pandas as pd
import numpy as np
import datetime
import tensorflow as tf
import numpy as np
import metrics
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

def list_dates(start = '2017-08-23', end = '2017-08-23', window_size=28):
    try:
        delta = (dt.datetime.strptime(end, '%Y-%m-%d') - dt.datetime.strptime(start, '%Y-%m-%d')).days
    except:
        delta = (end - start).days
    dates=[]
    for i in range(delta+window_size):
        try:
            dates.append(dt.datetime.strptime(start, '%Y-%m-%d') - dt.timedelta(days=window_size) + dt.timedelta(days=i))
        except:
            dates.append(start-dt.timedelta(days=window_size) + dt.timedelta(days=i))
    return np.asarray(dates)    
    
class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
def rescale(df, data):
    unscaled = pd.DataFrame(index=df.index)

    true = df['True'].values
    y_hat = df['Pred'].values


    if 'Std' in df.columns:
        std = df['Std'].values
        std = data.scaler.inverse_transform(np.tile(std+true, (255,1)).T)[:,-1]


    true =  data.scaler.inverse_transform(np.tile(true, (255,1)).T)[:,-1]
    y_hat =  data.scaler.inverse_transform(np.tile(y_hat, (255,1)).T)[:,-1]


    unscaled['True'] = true
    unscaled['Pred'] = y_hat

    if 'Std' in df.columns:
        std = std-true
        unscaled['Std'] = std
    
    return unscaled

def harmonic_smoothing(df, n=7):
    data = df.values
    for i in range(data.shape[0]):
        for j in range(data.shape[1]-1, n, -1):
            s = 1
            for k in range(1,n):
                data[i, j] = data[i,j] + data[i,j-n]/(k+1)
                s = s + 1/(k+1)
            data[i, j] = data[i,j]/s
            
    df = pd.DataFrame(columns = df.columns, index=df.index[-data.shape[0]:], data=data)
    return df

import datetime as dt
class data_builder_new:
    def __init__(self, root,country='eng'):
        self.directory = root

    def window(self, data, window_size=28):
        windowed = []
        for i in range(1+data.shape[0] - window_size):
            windowed.append(data[i:i + window_size])
        windowed = np.asarray(windowed)
        return windowed
    
    def build_2(self, test_season=2014, gamma=14, lag=14, window_size=28, smooth=True):
            weeks = int(gamma/7)
            season = str(test_season) + '/' + str(test_season+1)
            wk = pd.read_csv('../../../Datasets/Flu/wk'+str(weeks)+'.csv',index_col=0, parse_dates=True)
            wk = wk[wk['Season'] == season]
            y_test = pd.DataFrame(index=wk.index, columns=['ILI'], data=wk['Valid Bin_start_incl'].values)

            if smooth:
                Qs = pd.read_csv(self.directory + 'google_queries_pre_processed_us_smooth.csv', 
                                 index_col=0, 
                                 parse_dates=True)
            else:
                Qs = pd.read_csv(self.directory + 'google_queries_pre_processed_us.csv', 
                                 index_col=0, 
                                 parse_dates=True)
            
            ILI = pd.read_csv('/home/mimorris/Datasets/Flu/flusight_ili_gt.csv', 
                  parse_dates=True)
            ILI = ILI.set_index('date')
            ILI.index = pd.to_datetime(ILI.index)
            
            Qs = Qs[:ILI.index[-1]]
            ILI.index = ILI.index + dt.timedelta(days=lag)
            Qs['ILI'] = ILI.loc[Qs.index[0]:Qs.index[-1]]
            
            self.scaler = MinMaxScaler()
            self.scaler.fit(Qs)
            
            
            window_size=window_size-1
            
            Qs = pd.DataFrame(index=Qs.index, columns=Qs.columns, data=self.scaler.transform(Qs))
            
            train_index = list_dates(start=Qs.index[gamma+window_size], 
                                     end=y_test.index[0]+dt.timedelta(days=-64),
                                     window_size=window_size)
            
            test_index = list_dates(start=y_test.index[0], 
                         end=y_test.index[-1]+dt.timedelta(days=1),
                         window_size=window_size)
            x_train = Qs.loc[train_index - dt.timedelta(days=gamma)]
            x_test = Qs.loc[test_index - dt.timedelta(days=gamma)]
            
            y_train = Qs.loc[(train_index + dt.timedelta(days=lag))[window_size:]]['ILI']
            y_test_full = Qs.loc[(test_index + dt.timedelta(days=lag))[window_size:]]['ILI']
            
            x_train = self.window(x_train, window_size=window_size+1)
            x_test = self.window(x_test, window_size=window_size+1)
            
            
            return x_train, y_train, x_test, y_test, y_test_full

    def build(self, test_year=2017, validation_year=2016, start_of_season = '-08-23', gamma=14, lag=14, window_size=28, smooth=True):
            
            if smooth:
                Qs = pd.read_csv(self.directory + 'google_queries_pre_processed_us_smooth.csv', 
                                 index_col=0, 
                                 parse_dates=True)
            else:
                Qs = pd.read_csv(self.directory + 'google_queries_pre_processed_us.csv', 
                                 index_col=0, 
                                 parse_dates=True)
            
#             ILI = pd.read_csv('/home/mimorris/Datasets/Flu/ILI_rates_US_wednesday_linear_interpolation.csv', 
#                               parse_dates=True)
            ILI = pd.read_csv('/home/mimorris/Datasets/Flu/flusight_ili_gt.csv', 
                  parse_dates=True)
            ILI = ILI.set_index('date')
            ILI.index = pd.to_datetime(ILI.index)
            
            Qs = Qs[:ILI.index[-1]]
            ILI.index = ILI.index + dt.timedelta(days=lag)
            Qs['ILI'] = ILI.loc[Qs.index[0]:Qs.index[-1]]
            
            self.scaler = MinMaxScaler()
            self.scaler.fit(Qs)
            
            window_size=window_size-1
            
            Qs = pd.DataFrame(index=Qs.index, columns=Qs.columns, data=self.scaler.transform(Qs))
            
            validation_index = list_dates(start=str(validation_year)+start_of_season, 
                                          end=str(validation_year+1)+start_of_season, 
                                          window_size=window_size)
            
            test_index = list_dates(start=str(test_year)+start_of_season, 
                                    end=str(test_year+1)+start_of_season,
                                    window_size=window_size)
            
            train_index = list_dates(start=dt.datetime.strftime(Qs.index[gamma+window_size], '%Y-%m-%d'), 
                                     end=str(validation_year)+start_of_season,
                                     window_size=window_size)
            
            x_train = Qs.loc[train_index - dt.timedelta(days=gamma)]
            x_val = Qs.loc[validation_index - dt.timedelta(days=gamma)]
            x_test = Qs.loc[test_index - dt.timedelta(days=gamma)]
            
            y_train = Qs.loc[(train_index + dt.timedelta(days=lag))[window_size:]]['ILI']
            y_val = Qs.loc[(validation_index + dt.timedelta(days=lag))[window_size:]]['ILI']
            y_test = Qs.loc[(test_index + dt.timedelta(days=lag))[window_size:]]['ILI']
            
            x_train = self.window(x_train, window_size=window_size+1)
            x_val = self.window(x_val, window_size=window_size+1)
            x_test = self.window(x_test, window_size=window_size+1)
            
            
            return x_train, y_train, x_val, y_val, x_test, y_test
        
class results_table:
    def __init__(self, args):
        self.test_predictions = pd.DataFrame()
        self.test_metrics = pd.DataFrame(index = ['CRPS','NLL','MAE','RMSE','SMAPE','Corr','MB Log','SDP'])
        
        self.val_predictions = pd.DataFrame()
        self.val_metrics = pd.DataFrame(index = ['CRPS','NLL','MAE','RMSE','SMAPE','Corr','MB Log','SDP'])
        
        self.args = ['Arch='+str(args.Arch), 
                        'num_layers='+str(args.num_layers), 
                        'sizeof_layers='+str(args.sizeof_layers),
                        'batch_norm='+str(args.batch_norm),
                        'Ext='+str(args.Ext), 
                        'kl_anneal='+str(args.kl_anneal),
                        'rho_q='+str(args.rho_q),
                        'rho_op='+str(args.rho_op),
                        'prior_scale='+str(args.prior_scale),
                        'sizeof_bnn='+str(args.sizeof_bnn),
                        'Batch_Size='+str(args.Batch_Size), 
                        'Epochs='+str(args.Epochs), 
                        'Gamma='+str(args.Gamma), 
                        'country='+str(args.country), 
                        'early_stopping='+str(args.early_stopping), 
                        'smooth='+str(args.smooth)]
        
    def update(self, test_predictions, val_predictions=None, year=None):
        self.test_predictions = self.test_predictions.append(test_predictions)
        
        try:
            self.test_metrics[str(year + 2014) + '/' + str(year + 15)] = [metrics.crps(test_predictions),
                                                        metrics.nll(test_predictions),
                                                        metrics.mae(test_predictions),
                                                        metrics.rmse(test_predictions),
                                                        metrics.smape(test_predictions),
                                                        metrics.corr(test_predictions),
                                                        np.ma.masked_invalid(metrics.mb_log(test_predictions)).mean(),
                                                        metrics.sdp(test_predictions)]
        except:
            pass
        if year == 3:
            self.test_metrics['Average'] = self.test_metrics.mean(1)
            self.test_metrics['Average'].loc['SDP'] = np.abs(self.test_metrics.loc['SDP'].values[-1]).mean()
            
            
        try:
            self.val_predictions= self.val_predictions.append(val_predictions)
            self.val_metrics[str(year + 2013) + '/' + str(year + 14)] = [metrics.crps(val_predictions),
                                                            metrics.nll(val_predictions),
                                                            metrics.mae(val_predictions),
                                                            metrics.rmse(val_predictions),
                                                            metrics.smape(val_predictions),
                                                            metrics.corr(val_predictions),
                                                            metrics.mb_log(val_predictions).mean(),
                                                            metrics.sdp(val_predictions)]
        except:
            pass

            self.val_metrics['Average'] = self.val_metrics.mean(1)
            self.val_metrics['Average'].loc['SDP'] = np.abs(self.val_metrics.loc['SDP'].values[-1]).mean()
            self.test_metrics['Average'] = self.test_metrics.mean(1)
            self.test_metrics['Average'].loc['SDP'] = np.abs(self.test_metrics.loc['SDP'].values[-1]).mean()
    
    def save(self, dir, extension):
        root = os.getcwd()
        os.chdir(dir)
        
        os.mkdir(str(extension))
        os.chdir(str(extension))
        self.val_metrics.to_csv('val_metrics.csv')
        self.val_predictions.to_csv('val_predictions.csv')
        self.test_metrics.to_csv('test_metrics.csv')
        self.test_predictions.to_csv('test_predictions.csv')
        
        with open('args.txt', 'w') as f:
            for item in self.args:
                f.write("%s\n" % item)
        
        os.chdir(root)
        
def get_num(root):
    nums = []
    for i in np.asarray(os.listdir(root)):
        try:
            nums.append(int(i))
        except:
            pass
    return(np.asarray(nums).max()+1)