import pandas as pd
import numpy as np
import datetime

class data_builder:
    def __init__(self, args, fold):
        self.fold=fold
        self.args = args

        self.directory = args.Root + str(args.Gamma) + '/fold' + str(fold) + '/'

    def load_ili_data(self, path):
        ili_data = pd.read_csv(path, header=None, names=['ili'], index_col=0, parse_dates=True)
        return ili_data

    def load_google_data(self, path):
        google_data = pd.read_csv(path, index_col=0, parse_dates=True)
        return google_data

    def window(self, data):
        windowed = []
        for i in range(1+data.shape[0] - 28):
            windowed.append(data.iloc[i:i + 28].values)
        windowed = np.asarray(windowed)
        return windowed

    def mimo(self, y, ground_truth):
        data = []
        for i in range(y.shape[0]):
            start = y.index[i] - datetime.timedelta(0)
            end = y.index[i] + datetime.timedelta(1)
            data.append(ground_truth.iloc[np.argwhere(ground_truth.index == start)[0][0]:np.argwhere(ground_truth.index == end)[0][0]].values.squeeze())

        data = np.asarray(data)
        columns = []
        for i in range(1):
            if i > 0:
                add = '+'
            else:
                add = ''
            columns.append('T' + add + str(i))
        data = pd.DataFrame(data=data, columns=columns, index=y.index)
        return data



    def build(self, normalize_all=False):
        google_train = self.load_google_data(self.directory + 'google-train')
        google_test = self.load_google_data(self.directory + 'google-test')

        google_train['ili'] = self.load_ili_data(self.directory + 'ili-train')['ili'].values
        google_test['ili'] = self.load_ili_data(self.directory + 'ili-test')['ili'].values

        y_train = self.load_ili_data(self.directory + 'y-train')
        y_test = self.load_ili_data(self.directory + 'y-test')

        n = normalizer(google_train, y_train, normalize_all=normalize_all)

        google_train = n.normalize(google_train)
        google_test = n.normalize(google_test)

        x_train = self.window(google_train)
        x_test = self.window(google_test)

        y_train = y_train.rename(columns={'ili': 'T0'})
        y_test = y_test.rename(columns={'ili': 'T0'})

        assert (x_train.shape[0] == y_train.shape[0])
        assert (x_test.shape[0] == y_test.shape[0])

        # y_test = y_test.values
        y_train = y_train.values

        return x_train.astype('float32'), y_train.astype('float32'), x_test.astype('float32'), y_test.astype('float32')

class normalizer:
    def __init__(self, x, y, normalize_all=False):
        if normalize_all:
            self.x_min = np.min(np.asarray(x), axis=0)
            self.x_max = np.max(np.asarray(x), axis=0)

        else:
            self.x_min = np.min(np.asarray(x.iloc[:, :-1]), axis=0)
            self.x_max = np.max(np.asarray(x.iloc[:, :-1]), axis=0)

        self.normalize_all = normalize_all
        self.y_min = np.min(y)[0]
        self.y_max = np.max(y)[0]

    def normalize(self, X):
        if not self.normalize_all:
            x_val = np.asarray(X.iloc[:, :-1])
        else:
            x_val = np.asarray(X)

        for i in range(x_val.shape[0]):
            x_val[i] = (x_val[i] - self.x_min) / (self.x_max - self.x_min)

        if not self.normalize_all:
            x_val = np.concatenate([x_val, X.iloc[:, -1].values[:, np.newaxis]], 1)
        X_norm = pd.DataFrame(data=x_val, index = X.index, columns=X.columns)
        return X_norm

    def un_normalize(self, Y):
        y_val = np.asarray(Y[1])
        for i in range(y_val.shape[0]):
            y_val[i] = y_val[i] * (self.y_max - self.y_min) - self.y_min

        Y[1] = y_val
        return Y