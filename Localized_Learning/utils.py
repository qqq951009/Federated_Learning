import time
import threading
import pickle
import flwr as fl
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.under_sampling import RandomUnderSampler, NearMiss 
from category_encoders import TargetEncoder, LeaveOneOutEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import OneHotEncoder

# Use the data from 2011 and the data contain less than 9 null value counts
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

'''def imputation(df_train, df_test, imp_method, seed):
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
    return train_imp, test_imp'''


class target_encoding():
    def __call__(self, trainset, testset):
        columns = trainset.columns[2:]
        x_train, y_train = trainset.drop(columns=['Class', 'LOC']), trainset['Class']
        x_test, y_test = testset.drop(columns=['Class', 'LOC']), testset['Class']
        
        encoder = TargetEncoder(cols=columns, smoothing=0.05)
        encoder = encoder.fit(x_train, y_train)
        x_train_enc, x_test_enc = encoder.transform(x_train), encoder.transform(x_test)

        return x_train_enc, y_train, x_test_enc, y_test

class onehot_encoding():
    def __call__(self, trainimp, testimp):
        df = pd.concat([trainimp, testimp])
        df_x, df_y = df.drop(columns=['Class', 'LOC']), df['Class']
        x_enc = pd.get_dummies(df_x.astype(str), drop_first=True)
        x_train_enc = x_enc.loc[trainimp.index]
        x_test_enc = x_enc.loc[testimp.index]
        y_train = df_y.loc[trainimp.index]
        y_test = df_y.loc[testimp.index]
        return  x_train_enc, y_train, x_test_enc, y_test

def encode(trainset, testset, method):
    if method == 'onehot':
        df = pd.concat([trainset, testset])
        df_x, df_y = df.drop(columns=['Class', 'LOC']), df['Class']
        x_enc = pd.get_dummies(df_x.astype(str), drop_first=True)
        x_train_enc = x_enc.loc[trainset.index]
        x_test_enc = x_enc.loc[testset.index]
        y_train = df_y.loc[trainset.index]
        y_test = df_y.loc[testset.index]

    elif method == 'target':
        columns = trainset.columns[2:]
        x_train, y_train = trainset.drop(columns=['Class', 'LOC']), trainset['Class']
        x_test, y_test = testset.drop(columns=['Class', 'LOC']), testset['Class']
        encoder = TargetEncoder(cols=columns, smoothing=0.05)
        encoder = encoder.fit(x_train, y_train)
        x_train_enc, x_test_enc = encoder.transform(x_train), encoder.transform(x_test)

    else:
        assert method == 'target' or method == 'onehot'

    return x_train_enc, y_train, x_test_enc, y_test