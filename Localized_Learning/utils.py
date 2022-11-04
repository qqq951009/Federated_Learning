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

# Drop the year before 2010
class drop_year():
    def __call__(self, df):
        df['FullDate'] = df['FullDate'].astype('string')
        df['year'] = [int(x[:4]) for x in list(df['FullDate'])]
        index = df[df['year'] < 2010].index.tolist()
        df = df.iloc[~df.index.isin(index)]
        df = df.drop(columns = ['year', 'FullDate'])
        return df

# Drop the year before 2010 the paitent data is more than 9 null value
class drop_year_null():
    def __call__(self, df):
        df['FullDate'] = df['FullDate'].astype('string')
        df['year'] = [int(x[:4]) for x in list(df['FullDate'])]
        df['null_count'] = list(df.isna().sum(axis=1))
        year_index = df[df['year'] < 2010].index.tolist()
        null_index = df[df['null_count'] >= 9].index.tolist()
        df = df.iloc[~df.index.isin(year_index)]
        df = df.iloc[~df.index.isin(null_index)]
        df = df.drop(columns = ['year', 'FullDate', 'null_count'])
        return df

'''class imputation():
    def __call__(self, df_train, df_test, imp_method):
        #if 'FullDate' not in df_train.columns:
        #    return df_train, df_test
        
        if imp_method == 'fill10':
            train_imp, test_imp = df_train.fillna(10), df_test.fillna(10)
        if imp_method == 'drop_and_fill':
            
            df_train['null_count'], df_test['null_count'] = list(df_train.isna().sum(axis=1)), list(df_test.isna().sum(axis=1))
            train_index = df_train[df_train['null_count'] >= 9].index.tolist()
            test_index = df_test[df_test['null_count'] >= 9].index.tolist()
            print(len(train_index), len(test_index))
            df_train = df_train.iloc[~df_train.index.isin(train_index)]
            df_test = df_test.iloc[~df_test.index.isin(test_index)]
            
            temp = pd.concat([df_train, df_test])
            print(temp.isna().sum(axis=1).value_counts())
            
            train_imp = df_train.fillna(df_train.median())
            test_imp = df_test.fillna(df_train.median())

            train_imp, test_imp = train_imp.drop(columns = ['null_count']), test_imp.drop(columns = ['null_count'])
            train_imp, test_imp = train_imp.astype(int), test_imp.astype(int)
        return train_imp, test_imp'''

# Drop the paitent data has more than 9 null value 
# And fill median to the remaining null value
'''class imputation():
    def __call__(self, df_train, df_test, imp_method):      
        if imp_method == '10':
            train_imp, test_imp = df_train.fillna(10), df_test.fillna(10)

        if imp_method == 'median':     
            train_imp = df_train.fillna(df_train.median())
            test_imp = df_test.fillna(df_train.median())

        #train_imp, test_imp = train_imp.drop(columns = ['null_count']), test_imp.drop(columns = ['null_count'])
        train_imp, test_imp = train_imp.astype(int), test_imp.astype(int)
        return train_imp, test_imp'''

class imputation():
    def __call__(self, df_train, df_test, imp_method):
        #if 'FullDate' not in df_train.columns:
        #    return df_train, df_test
        df_train['null_count'], df_test['null_count'] = list(df_train.isna().sum(axis=1)), list(df_test.isna().sum(axis=1))
        train_index = df_train[df_train['null_count'] >= 9].index.tolist()
        test_index = df_test[df_test['null_count'] >= 9].index.tolist()
        df_train = df_train.iloc[~df_train.index.isin(train_index)]
        df_test = df_test.iloc[~df_test.index.isin(test_index)]
        if imp_method == 'fill10':
            train_imp, test_imp = df_train.fillna(10), df_test.fillna(10)

        if imp_method == 'drop_and_fill':     
            train_imp = df_train.fillna(df_train.median())
            test_imp = df_test.fillna(df_train.median())

        train_imp, test_imp = train_imp.drop(columns = ['null_count']), test_imp.drop(columns = ['null_count'])
        train_imp, test_imp = train_imp.astype(int), test_imp.astype(int)
        return train_imp, test_imp

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
    def __call__(self, dict, df, columns) :
        for i in columns:
            df[i] = df[i].apply(lambda x:dict[i][x])
        return df

class split_data():    
    def __call__(self, df, size):
        pivot = int(len(df)*(1-size))
        trainset, testset = df[:pivot], df[pivot:]
        x_train, y_train = trainset.drop(columns = ['Class', 'LOC']), trainset['Class']
        x_test, y_test = testset.drop(columns = ['Class', 'LOC']), testset['Class']
        
        return x_train, x_test, y_train, y_test
