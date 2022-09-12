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

class imputation():
    def __call__(self, df, imp_method):
        if 'FullDate' not in df.columns:
            return df
        else:
            if imp_method == 'fill9':
                df_imp = df.fillna(9)
            if imp_method == 'drop':
                df_imp = df.dropna()
            if imp_method == 'drop_and_fill':
                df['year'] = [int(x[:4]) for x in list(df['FullDate'])]
                df['null_count'] = list(df.isna().sum(axis=1))
                df_imp = df.reset_index(drop = True)
                index = [i.tolist() for i in np.where( (df_imp['null_count']  >= 9) & (df_imp['year'] <= 2010))][0]
                df_imp = df_imp.iloc[~df_imp.index.isin(index)]
                df_imp = df_imp.fillna(df_imp.median())
                df_imp = df_imp.drop(columns = ['year','null_count','FullDate'])
                df_imp = df_imp.astype(int)
            return df_imp

class target_encoding():
    def __init__(self, loo: bool) :
        self.loo = loo

    def __call__(self, df):
        columns = df.columns[2:]
        x = df.drop(columns = ['Class'])
        y = df['Class']
        # choose leave one out
        if self.loo == True:
            encoder = LeaveOneOutEncoder(cols=columns ,sigma = 0.05)
        if self.loo == False:
            encoder = TargetEncoder(cols=columns,smoothing=0.05)
        
        df_target = encoder.fit_transform(x,y)
        df_target['Class'] = y
        return df_target

class choose_dict():
    def __init__(self, weight_average, seer):
        self.weight_average = weight_average
        self.seer = seer

    def __call__(self):
        if self.seer == 1:
            if self.weight_average == 'average':
                with open('/home/refu0917/lungcancer/remote_output1/encode_dict_folder/encode_average_seer.pickle', 'rb') as f:
                    return pickle.load(f)
            
            elif self.weight_average == 'weight':
                with open('/home/refu0917/lungcancer/remote_output1/encode_dict_folder/encode_weight.pickle_seer', 'rb') as f:
                    return pickle.load(f)
        elif self.seer == 0:
            if self.weight_average == 'average':
                with open('/home/refu0917/lungcancer/remote_output1/encode_dict_folder/encode_average.pickle', 'rb') as f:
                    return pickle.load(f)
            
            elif self.weight_average == 'weight':
                with open('/home/refu0917/lungcancer/remote_output1/encode_dict_folder/encode_weight.pickle', 'rb') as f:
                    return pickle.load(f)

class mapping():
    def __call__(self, dict, df) :
        for i in df.columns[2:]:
            df[i] = df[i].apply(lambda x:dict[i][x])
        return df

class split_data():
    def __init__(self, testsize, seed):
        self.testsize = testsize
        self.seed = seed
    
    def __call__(self, df):
        df = df.drop(columns = ['LOC'])
        trainset,testset = train_test_split(df,test_size = self.testsize,stratify=df['Class'],random_state=self.seed)
        x_train = trainset.drop(columns=['Class'])
        x_test = testset.drop(columns=['Class'])
        y_train = trainset['Class']
        y_test = testset['Class']
        return x_train, x_test, y_train, y_test 

class sample_method():
    def __init__(self,sampler,strategy,seed):
        self.sampler = sampler
        self.seed = seed
        self.strategy = strategy

    '''class foo(random_state, sampling_strategy, method="smotetomek"):
        method = ''
        def __init__():
        
        def execute():
            getattr(self, SMOTEENN)(seed, strategy)'''

    def __call__(self,x_train,y_train):
        #foo = foo(self.sedd, self.strategy, "smotetomek")
        #foo.execute()
        
        if self.sampler == 'smotetomek':
            sample_method = SMOTETomek(random_state=self.seed, sampling_strategy = self.strategy)
        if self.sampler == 'smoteenn':
            sample_method = SMOTEENN(random_state=self.seed, sampling_strategy = self.strategy)
        if self.sampler == 'undersample':
            sample_method = RandomUnderSampler(random_state=self.seed, sampling_strategy = self.strategy)
        if self.sampler == 'nearmiss':
            sample_method = NearMiss(random_state=self.seed, sampling_strategy = self.strategy)
        
        x_train_smote,y_train_smote = sample_method.fit_resample(x_train, y_train)
        return x_train_smote,y_train_smote

