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

class imputation():
    def __call__(self, df_train, df_test, imp_method):
        #if 'FullDate' not in df_train.columns:
        #    return df_train, df_test
        
        if imp_method == 'fill10':
            train_imp, test_imp = df_train.fillna(10), df_test.fillna(10)
        if imp_method == 'drop_and_fill':
            
            df_train['null_count'], df_test['null_count'] = list(df_train.isna().sum(axis=1)), list(df_test.isna().sum(axis=1))
            train_index = df_train[df_train['null_count'] >= 9].index.tolist()
            test_index = df_test[df_test['null_count'] >= 9].index.tolist()
            df_train = df_train.iloc[~df_train.index.isin(train_index)]
            df_test = df_test.iloc[~df_test.index.isin(test_index)]
            
            train_imp = df_train.fillna(df_train.median())
            test_imp = df_test.fillna(df_train.median())

            train_imp, test_imp = train_imp.drop(columns = ['null_count']), test_imp.drop(columns = ['null_count'])
            train_imp, test_imp = train_imp.astype(int), test_imp.astype(int)
        return train_imp, test_imp

class drop_year():
    def __call__(self, df):
        df['FullDate'] = df['FullDate'].astype('string')
        df['year'] = [int(x[:4]) for x in list(df['FullDate'])]
        index = df[df['year'] < 2010].index.tolist()
        df = df.iloc[~df.index.isin(index)]
        df = df.drop(columns = ['year', 'FullDate'])
        return df

class iterative_imputation():
    def __call__(self,df,seed):
        imputer = IterativeImputer(random_state=seed, estimator=RandomForestClassifier(),initial_strategy = 'most_frequent')
        df_imp = imputer.fit_transform(df)
        df_imp = df_imp.astype(int)
        df_imp = pd.DataFrame(data = df_imp,columns = df.columns)
        return df_imp

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

class train_enc_map():
    def __call__(self, dfenc, dfimp, columns,df):
        trainenc_dict = {}
        for col in columns:
            trainenc_dict[col] = dict((int(key),0) for key in df[col].value_counts().index.tolist())
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
        

    def __call__(self, df_list, index_list, df, columns):
        map, enc_dict, imp_dict = {}, {}, {}

        drop_year_fn = drop_year()
        target_encode_fn = target_encoding(False)
        imputation_fn = imputation()
        # iterative_imputation_fn = iterative_imputation()
        train_enc_map_fn = train_enc_map()
        
        for i, site_id in zip(range(len(df_list)), index_list):                     
            temp = drop_year_fn(df_list[i])
            trainset, testset = train_test_split(temp,test_size = self.size,stratify=temp['Class'],random_state=self.seed)
            trainimp, testimp = imputation_fn(trainset, testset, 'drop_and_fill')
            
            # Make train and test imputation dictionary
            imp_dict[site_id] = {"train":trainimp, "test":testimp}

            # Traget encode trainset and make trainset encode dictionary
            trainenc = target_encode_fn(trainimp)
            df_list[i] = train_enc_map_fn(trainenc,trainimp, columns[3:],df)

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
