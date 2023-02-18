import os
import yaml
import time
import pickle
import collections
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

# Use the data from 2011 and the data contain less than 9 null value counts
class drop_year_and_null():
    def __call__(self, df):
        df['FullDate'] = df['FullDate'].astype('string')
        df['year'] = [int(x[:4]) for x in list(df['FullDate'])]
        df['null_count'] = list(df.isna().sum(axis=1))
        year_index = df[df['year'] < 2011].index.tolist()
        null_index = df[df['null_count'] >= 9].index.tolist()
        df = df.iloc[~df.index.isin(year_index)]
        df = df.iloc[~df.index.isin(null_index)]
        df = df.drop(columns = ['year', 'FullDate', 'null_count'])
        return df

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
    def __init__(self, loo: bool) :
        self.loo = loo

    def __call__(self, df):
        columns = df.columns[2:]
        loc = df['LOC']
        x = df.drop(columns = ['Class','LOC'])
        y = df['Class']
        # choose leave one out
        if self.loo == True:
            encoder = LeaveOneOutEncoder(cols=columns ,sigma = 0.05)
        if self.loo == False:
            encoder = TargetEncoder(cols=columns,smoothing=0.05)
        
        df_target = encoder.fit_transform(x,y)
        df_target['Class'] = y
        df_target['LOC'] = loc
        return df_target

class onehot_encoding():
    def __call__(self, df, siteid, train_index, test_index):
        trainset, testset = [], []
        df = pd.get_dummies(df.astype(str), drop_first=True, columns=df.columns[2:])
        df = df[df['LOC'] == str(siteid)]
        trainset = df.loc[train_index]
        testset = df.loc[test_index]
        trainset = trainset.astype(int)
        testset = testset.astype(int)
        x_train, y_train = trainset.drop(columns=['Class', 'LOC']), trainset['Class']
        x_test, y_test = testset.drop(columns=['Class', 'LOC']), testset['Class']
        return x_train, y_train, x_test, y_test

class train_enc_map():
    def __call__(self, dfenc, dfimp, columns,df):
        trainenc_dict = {}
        for col in columns:
            trainenc_dict[col] = dict((int(key),0) for key in df[col].value_counts().index.tolist())
            trainenc_dict[col][10] = 0
            implist = dfimp[col].value_counts().index.tolist()
            for i in implist:
                id = dfimp.loc[dfimp[col] == i].index[0]
                trainenc_dict[col][i] = dfenc.loc[id, col]           
        return trainenc_dict

class mapping():
    def __call__(self, dict, df, col_list) :
      
        for i in col_list:
            df[i] = df[i].apply(lambda x:dict[i][x])
        return df
 
class make_map():
    def __init__(self, seed, size):
        self.seed = seed
        #self.seer = seer
        self.size = size
        

    def __call__(self, df_list, index_list, df, columns, imp_method):
        map, enc_dict, imp_dict = {}, {}, {}

        # drop_fn = drop_year_and_null()
        target_encode_fn = target_encoding(False)
        imputation_fn = imputation()
        # iterative_imputation_fn = iterative_imputation()
        train_enc_map_fn = train_enc_map()
        
        for i, site_id in zip(range(len(df_list)), index_list):                     
            # temp = drop_fn(df_list[i])
            trainset, testset = train_test_split(df_list[i], test_size = self.size,stratify = df_list[i]['Class'], random_state=self.seed)
            trainimp, testimp = imputation_fn(trainset, testset, imp_method, self.seed)
            
            # Make train and test imputation dictionary
            imp_dict[site_id] = {"train":trainimp, "test":testimp}

            # Traget encode trainset and make trainset encode dictionary
            trainenc = target_encode_fn(trainimp)
            df_list[i] = train_enc_map_fn(trainenc,trainimp, columns[2:],df)

        for k in df_list[0].keys():
            temp_list = []
            for s in range(len(df_list)):
                temp_list.append(df_list[s][k])
            counter = collections.Counter()
            for ele in temp_list:
                counter.update(ele)
            res = dict(counter)
            for key , value in zip(res.keys(), res.values()):
                res[key] = value/len(df_list)
            map[k] = res
        
        with open('mapping.pickle', 'wb') as f:
            pickle.dump(map, f)
        with open('imputationdf.pickle', 'wb') as f:
            pickle.dump(imp_dict, f)
            

class split_data():    
    def __call__(self, df, size):
        pivot = int(len(df)*(1-size))
        trainset, testset = df[:pivot], df[pivot:]
        x_train, y_train = trainset.drop(columns = ['Class', 'LOC']), trainset['Class']
        x_test, y_test = testset.drop(columns = ['Class', 'LOC']), testset['Class']
        
        return x_train, x_test, y_train, y_test

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

class pretrain_site():
    def __call__(self,df_list, index_list):
        max_len = 0
        ptr_site = 0
        for df, site_id in zip(df_list, index_list):
            if len(df) > max_len:
                max_len = len(df)
                ptr_site = site_id
        return ptr_site
