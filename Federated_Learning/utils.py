import time
import collections
import pickle
import flwr as fl
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import metrics
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.under_sampling import RandomUnderSampler, NearMiss 
from category_encoders import TargetEncoder, LeaveOneOutEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from tensorflow.keras.optimizers import Adam

# Drop the year before 2010 the paitent data is more than 9 null value
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
            trainset, testset = train_test_split(df_list[i], test_size = self.size, stratify = df_list[i]['Class'], random_state=self.seed)
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
        with open('./encode_dict_folder/mapping.pickle', 'wb') as f:
            pickle.dump(map, f)
        with open('./encode_dict_folder/imputationdf.pickle', 'wb') as f:
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


class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test, cid, size, seed, seer, client_config):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.cid = cid
        self.record = []
        self.auc_val_result = {}
        self.set_thres = 0.19
        self.seed = seed
        self.size = size
        if seer == 1:
            self.hospital_list = [2,3,6,8,9]
            self.output_file_name = '/home/refu0917/lungcancer/remote_output1/output_folder/iterative_impute_folder/df_fedavg_average_seer'
        elif seer == 0:
            self.hospital_list = [2,3,6,8,9,10,11,12]
            self.output_file_name = client_config['dir_name']+'df_fedavg_average'
        self.METRICS = [metrics.Precision(thresholds=self.set_thres),
                    metrics.Recall(thresholds=self.set_thres),
                    metrics.AUC()]
        self.opt_adam = Adam(learning_rate=0.001)         
    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""
    
        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]
    
        # Train the model using hyperparameters from config
        self.model.compile(optimizer=self.opt_adam, loss=tf.losses.BinaryFocalCrossentropy(gamma=2.0), metrics=self.METRICS)
        history = self.model.fit(self.x_train, self.y_train, batch_size, epochs,validation_data=(self.x_test, self.y_test))
        
        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        hist_key_list = list(history.history.keys())
        results = {
            "loss": history.history[hist_key_list[0]][0],
            "auc": history.history[hist_key_list[3]][0],
            "val_loss": history.history[hist_key_list[4]][0],
            "val_auc": history.history[hist_key_list[7]][0],
        }
        # self.record+=history.history["val_auc"]
        print(config['rnd'])
        # if config['rnd'] == 20:
        #     print("output length : ",len(self.record))
        #     fl_auc_df = pd.read_csv(f'/home/refu0917/lungcancer/remote_output1/output_folder/fl_auc_df{self.cid}.csv') 
        #     fl_auc_df[f'seed{self.seed}'] = self.record
        #     fl_auc_df.to_csv(f'/home/refu0917/lungcancer/remote_output1/output_folder/fl_auc_df{self.cid}.csv',index=False)
            
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Use aggregate model to predict test data
        y_pred = self.model.predict(self.x_test)
        
        # Last round drawlift chart and Evaluate aggregate model on other hospital
        if config['rnd'] == 5:
            
        # Start evaluate each hospital validation set
            self.auc_val_result[str(self.cid)] = [roc_auc_score(self.y_test,y_pred)]            # Last round result 
            self.hospital_list.remove(self.cid)
            
            y_test_pred = self.model.predict(self.x_test)
            auroc = roc_auc_score(self.y_test,y_test_pred)
            
            print("----------------evaluate---------------")
            print(self.cid, auroc)
            '''val_df = pd.read_csv(f'{self.output_file_name}{self.cid}.csv', index_col=[0])
            val_df.loc[self.seed,'site'+str(self.cid)] =  auroc
            val_df.to_csv(f'{self.output_file_name}{self.cid}.csv')'''

        # Evaluate global model parameters on the local test data and return results
        loss,precision,recall,_ = self.model.evaluate(self.x_test, self.y_test)
        results = {"loss" : loss, "auc":roc_auc_score(self.y_test,y_pred)}

        num_examples_test = len(self.x_test)
        return loss, num_examples_test, results




class CifarClient_personal(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test, cid, size, seed, seer, client_config):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.cid = cid
        self.record = []
        self.auc_val_result = {}
        self.set_thres = 0.19
        self.seed = seed
        self.size = size
        if seer == 1:
            self.hospital_list = [2,3,6,8,9]
            self.output_file_name = '/home/refu0917/lungcancer/remote_output1/output_folder/iterative_impute_folder/df_fedavg_average_seer'
        elif seer == 0:
            self.hospital_list = [2,3,6,8,9,10,11,12]
            self.output_file_name = client_config['dir_name']+'df_fedavg_average'
        self.METRICS = [metrics.Precision(thresholds=self.set_thres),
                    metrics.Recall(thresholds=self.set_thres),
                    metrics.AUC()]
        self.opt_adam = Adam(learning_rate=0.001)         
    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""
    
        # Update local model parameters
        #　self.model.set_weights(parameters)
        for layer_name in ['base1', 'base2']:   # , 'base3'
            start_index = (int(layer_name[-1])-1)*2
            end_index = int(layer_name[-1])*2
            self.model.get_layer(layer_name).set_weights(parameters[start_index:end_index])
        
        # basic_param_before = self.model.get_layer('base3').get_weights()
        # personal_layer = self.model.get_layer('personal').get_weights()
        # round = config['rnd']
        # print(f'Round{round} site {self.cid} peronal parameter : {personal_layer}')

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]
    
        # Train the model using hyperparameters from config
        self.model.compile(optimizer=self.opt_adam, loss=tf.losses.BinaryFocalCrossentropy(gamma=2.0), metrics=self.METRICS)
        history = self.model.fit(self.x_train, self.y_train, batch_size, epochs,validation_data=(self.x_test, self.y_test))
        
        # basic_param_after = self.model.get_layer('base3').get_weights()
        # print(f'Round{round} site {self.cid} basic3 parameter before train : {basic_param_before[0][:2]}')
        # print(f'Round{round} site {self.cid} basic3 parameter after train : {basic_param_after[0][:2]}')
        # print(f'Round{round} site {self.cid} basic layer parameter diff: {np.array(basic_param_after[0][:2])-np.array(basic_param_before[0][:2])}')

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()[:4]
        num_examples_train = len(self.x_train)
        hist_key_list = list(history.history.keys())
        results = {
            "loss": history.history[hist_key_list[0]][0],
            "auc": history.history[hist_key_list[3]][0],
            "val_loss": history.history[hist_key_list[4]][0],
            "val_auc": history.history[hist_key_list[7]][0],
        }
        print(config['rnd'])
      
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        # self.model.set_weights(parameters)
        for layer_name in ['base1', 'base2']:   # , 'base3'
            start_index = (int(layer_name[-1])-1)*2
            end_index = int(layer_name[-1])*2
            self.model.get_layer(layer_name).set_weights(parameters[start_index:end_index])

        # Use aggregate model to predict test data
        y_pred = self.model.predict(self.x_test)
        
        # Last round drawlift chart and Evaluate aggregate model on other hospital
        if config['rnd'] == 3:
            
        # Start evaluate each hospital validation set
            self.auc_val_result[str(self.cid)] = [roc_auc_score(self.y_test,y_pred)]            # Last round result 
            self.hospital_list.remove(self.cid)
            
            y_test_pred = self.model.predict(self.x_test)
            auroc = roc_auc_score(self.y_test,y_test_pred)
            
            print("----------------evaluate---------------")
            print(self.cid, auroc)
            val_df = pd.read_csv(f'{self.output_file_name}{self.cid}.csv', index_col=[0])
            val_df.loc[self.seed,'site'+str(self.cid)] =  auroc
            val_df.to_csv(f'{self.output_file_name}{self.cid}.csv')
            
        # Evaluate global model parameters on the local test data and return results
        loss,precision,recall,_ = self.model.evaluate(self.x_test, self.y_test)
        results = {"loss" : loss, "auc":roc_auc_score(self.y_test,y_pred)}

        num_examples_test = len(self.x_test)
        return loss, num_examples_test, results