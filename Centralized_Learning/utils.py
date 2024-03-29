import time
import threading
import pickle
import flwr as fl
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.under_sampling import RandomUnderSampler, NearMiss 
from category_encoders import TargetEncoder, LeaveOneOutEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier



# Drop the year before 2010 the paitent data is more than 9 null value
class drop_year_and_null():
    def __call__(self, df):
        df['FullDate'] = df['FullDate'].astype('string')
        df['year'] = [int(x[:4]) for x in list(df['FullDate'])]
        df['null_count'] = list(df.isna().sum(axis=1))
        year_index = df[df['year'] < 2011].index.tolist()
        null_index = df[df['null_count'] >= 9].index.tolist()
        df_dropyear = df.iloc[~df.index.isin(year_index)]
        df_dropnull = df_dropyear.iloc[~df_dropyear.index.isin(null_index)]
        df_dropnull = df_dropnull.drop(columns = ['year', 'FullDate', 'null_count'])
        return df_dropnull

class imputation():
    def __call__(self, df_train, df_test, imp_method, seed):        
        if imp_method == '10':
            train_imp, test_imp = df_train.fillna(10), df_test.fillna(10)

        if imp_method == 'median':
            train_imp = df_train.fillna(df_train.median())
            test_imp = df_test.fillna(df_train.median())

        if imp_method == 'iterative':
            imputer = IterativeImputer(random_state=seed, estimator=RandomForestClassifier(),initial_strategy = 'most_frequent')
            imputer = imputer.fit(df_train)
            trainimp = imputer.transform(df_train)
            testimp = imputer.transform(df_test)
            train_imp, test_imp = pd.DataFrame(data=trainimp, columns=df_train.columns), pd.DataFrame(data=testimp, columns=df_test.columns)
        
        train_imp, test_imp = train_imp.astype(int), test_imp.astype(int)
        return train_imp, test_imp


class target_encoding():
    def __call__(self, trainset, testset):
        columns = trainset.columns[2:]
        x_train, y_train = trainset.drop(columns=['Class', 'LOC']), trainset['Class']
        x_test, y_test = testset.drop(columns=['Class', 'LOC']), testset['Class']
        
        encoder = TargetEncoder(cols=columns, smoothing=0.05)
        encoder = encoder.fit(x_train, y_train)
        x_train_enc, x_test_enc = encoder.transform(x_train), encoder.transform(x_test)
        x_test_enc['LOC'] = testset['LOC']
        x_test_enc['Class'] = y_test
        return x_train_enc, y_train, x_test_enc

class onehot_encoding():
    def __call__(self, trainimp, testimp):
        df = pd.concat([trainimp, testimp])
        df_x, df_y = df.drop(columns=['Class', 'LOC']), df['Class']
        x_enc = pd.get_dummies(df_x.astype(str), drop_first=True)
        x_train = x_enc.loc[trainimp.index]
        testset = x_enc.loc[testimp.index]
        y_train = df_y.loc[trainimp.index]
        testset['Class'] = df_y.loc[testimp.index]
        testset['LOC'] = df.loc[testimp.index]['LOC']
        return  x_train, y_train, testset

def encode(trainset, testset, method):
    if method == 'onehot':
        df = pd.concat([trainset, testset])
        df_x, df_y = df.drop(columns=['Class', 'LOC']), df['Class']
        x_enc = pd.get_dummies(df_x.astype(str), drop_first=True)
        x_train_enc = x_enc.loc[trainset.index]
        testset_enc = x_enc.loc[testset.index]
        y_train = df_y.loc[trainset.index]
        testset_enc['Class'] = df_y.loc[testset.index]
        testset_enc['LOC'] = df.loc[testset.index]['LOC']

    elif method == 'target':
        columns = trainset.columns[2:]
        x_train, y_train = trainset.drop(columns=['Class', 'LOC']), trainset['Class']
        x_test, y_test = testset.drop(columns=['Class', 'LOC']), testset['Class']
        encoder = TargetEncoder(cols=columns, smoothing=0.05)
        encoder = encoder.fit(x_train, y_train)
        x_train_enc, testset_enc = encoder.transform(x_train), encoder.transform(x_test)
        testset_enc['LOC'] = testset['LOC']
        testset_enc['Class'] = y_test
    else:
        assert method == 'target' or method == 'onehot'

    return x_train_enc, y_train, testset_enc

class sample_method():
    def __init__(self,method,strategy,seed):
        self.method = method
        self.seed = seed
        self.strategy = strategy

    def __call__(self,x_train,y_train):
        sampler = self.sampler(self.seed, self.strategy, self.method)
        sample = sampler.execute()
        x_train_sample, y_train_sample = sample.fit_resample(x_train, y_train)
        
        return x_train_sample, y_train_sample

    class sampler():
        def __init__(self, seed, sampling_strategy, method):
            self.method = method
            self.RandomUnderSampler = RandomUnderSampler(random_state=seed, sampling_strategy = sampling_strategy)
            self.SMOTEENN = SMOTEENN(random_state=seed, sampling_strategy = sampling_strategy)  
            self.NearMiss = NearMiss(sampling_strategy = sampling_strategy)
            
        def execute(self):
            return getattr(self, self.method)


class preprocess():
    def __init__(self, size, seed):
        self.size = size
        self.seed = seed
    def __call__(self,df, site_list):
        train = pd.DataFrame()
        test = pd.DataFrame()
        df = df.reset_index(drop = True)
        for i in site_list:
            df_site = df[df['LOC'] == i]
            trainset, testset = train_test_split(df_site,test_size = self.size,stratify=df_site['Class'],random_state=self.seed)
            train = pd.concat([train, trainset])
            test = pd.concat([test, testset])
        return train, test