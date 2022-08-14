import os
import pickle
import random
import argparse
import flwr as fl
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import metrics
import scikitplot as skplt
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
from sklearn.utils import shuffle
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten,Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_auc_score
from category_encoders import TargetEncoder, LeaveOneOutEncoder

# Parse command line argument `partition`
parser = argparse.ArgumentParser(description="Flower")
parser.add_argument("--hospital", type=int, choices=range(0, 10), required=True)
parser.add_argument("--seed", type=int, choices=range(0, 1000), required=True)
args = parser.parse_args()

cid = args.hospital
seed = args.seed
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def imputation(df,method):
  if method == 'fill9':
    df = df.fillna(9)
  elif method == 'drop':
    df = df.dropna()
  for i in df.columns:
    df[i] = df[i].astype(int)
  return df

def drop_and_fill(df):
  df['year'] = [int(x[:4]) for x in list(df['FullDate'])]
  df['null_count'] = list(df.isna().sum(axis=1))
  df = df.reset_index(drop = True)
  index = [i.tolist() for i in np.where( (df['null_count']  >= 9) & (df['year'] <= 2010))][0]
  df = df.iloc[~df.index.isin(index)]
  df = df.fillna(df.median())
  df = df.drop(columns = ['year','null_count','FullDate'])
  df = df.astype(int)
  return df

# Target encoding leave one out
def target_encoding_loo(df):
  columns = df.columns[2:]
  x = df.drop(columns = ['Class'])
  y = df['Class']
  #encoder = LeaveOneOutEncoder(cols=columns ,sigma = 0.05)
  encoder = TargetEncoder(cols=columns,smoothing=0.05)
  df_target = encoder.fit_transform(x,y)
  df_target['Class'] = y
  return df_target

# SMOTE oversampling
def smote(x_train,y_train,sampling_strategy):
    smote = SMOTE(random_state=seed,sampling_strategy=sampling_strategy)
    smote_tomek = SMOTETomek(random_state = seed,sampling_strategy = sampling_strategy)
    smote_enn = SMOTEENN(random_state = seed,sampling_strategy = sampling_strategy)

    #x_train_smote,y_train_smote = smote.fit_resample(x_train,y_train)
    #x_train_smote,y_train_smote = smote_tomek.fit_resample(x_train,y_train)
    x_train_smote,y_train_smote = smote_enn.fit_resample(x_train,y_train)
    return x_train_smote, y_train_smote

def split_data(df,testsize,seed):
    df = df.drop(columns=['LOC'])
    trainset,testset = train_test_split(df,test_size = testsize,stratify=df['Class'],random_state=seed)
    x_train = trainset.drop(columns=['Class'])
    x_test = testset.drop(columns=['Class'])
    y_train = trainset['Class']
    y_test = testset['Class']

    # Oversampling
    #x_train,y_train = smote(x_train,y_train,0.5)
    #x_train,y_train = shuffle(x_train,y_train)

    return x_train,x_test,y_train ,y_test 

def load_data(hospital_id: int):
    assert hospital_id in range(10)
    df = pd.read_csv(r'/home/refu0917/lungcancer/server/AllCaseCtrl_final.csv')
    df = df[["Class","LOC", "FullDate","Gender", "Age", "CIG", "ALC", "BN",    #"FullDate",
            "MAGN", "AJCCstage", "DIFF", "LYMND", "TMRSZ",
            "OP", "RTDATE", "STDATE", "BMI_label",
            "SSF1", "SSF2", "SSF3", "SSF4", "SSF6"]]
    df['Class'] = df['Class'].apply(lambda x:1 if x != 0 else 0)
    df = df[df['LOC'] == hospital_id]

    # Ddata preprocess 
    #df = imputation(df,'fill9')
    df = drop_and_fill(df)
    df = target_encoding_loo(df)
    x_train,x_test,y_train,y_test = split_data(df,0.2,seed)
    return x_train,x_test,y_train,y_test



# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test, cid, record):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.cid = cid
        self.record = record
        self.auc_val_result = {}
        self.hospital_list = [2,3,6,8]

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
        history = self.model.fit(self.x_train, self.y_train, batch_size, epochs,validation_data=(self.x_test, self.y_test))
            
        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            "auc": history.history["auc"][0],
            "val_loss": history.history["val_loss"][0],
            "val_auc": history.history["val_auc"][0],
        }
        self.record+=history.history["val_auc"]

        if config['rnd'] == 20:
            fl_auc_df = pd.read_csv(f'./data_folder/fl_auc_df{self.cid}.csv')
            fl_auc_df[f'seed{seed}'] = self.record
            fl_auc_df.to_csv(f'./data_folder/fl_auc_df{self.cid}.csv',index=False)
            '''fl_auc_df = pd.DataFrame(data = self.record,columns = [f'seed{seed}'])
            fl_auc_df.to_csv(f'fl_df{self.cid}.csv',index=False)'''

            with open('FL_AUC_'+str(self.cid)+'.pickle', 'wb') as f:
                pickle.dump(self.record, f)

        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Use aggregate model to predict test data
        y_pred = self.model.predict(self.x_test)
           
        # Last round drawlift chart and Evaluate aggregate model on other hospital
        if config['rnd'] == 20:
        # Start evaluate each hospital validation set
            self.auc_val_result[str(self.cid)] = [roc_auc_score(self.y_test,y_pred)]            # Last round result 
            
            '''val_df = pd.read_csv('fl_val_df.csv', index_col=[0])
            val_df.loc[seed,f'site{self.cid}'] = roc_auc_score(self.y_test,y_pred)
            val_df.to_csv('fl_val_df.csv')'''

            self.hospital_list.remove(self.cid)
            for i in self.hospital_list:                                           # Loop each hospital to evaluate result
                _,x_val,_,y_val= load_data(i)
                y_val_pred = self.model.predict(x_val)
                self.auc_val_result[str(i)] = [roc_auc_score(y_val,y_val_pred)]
            self.auc_val_result = dict(sorted(self.auc_val_result.items()))         # Sort the dict
            with open('FL_AUC_val_'+str(self.cid)+'.pickle', 'wb') as f:
                pickle.dump(self.auc_val_result, f)
            if self.cid==2:
                val_df = pd.read_csv('./data_folder/df_fedadagrad.csv', index_col=[0])
                val_df.loc[seed] =  [i[0] for i in self.auc_val_result.values()]
                val_df.to_csv('./data_folder/df_fedadagrad.csv')
        # Evaluate global model parameters on the local test data and return results
        loss,precision,recall,_ = self.model.evaluate(self.x_test, self.y_test)
        results = {"loss" : loss, "auc":roc_auc_score(self.y_test,y_pred)}

        num_examples_test = len(self.x_test)
        return loss, num_examples_test, results


def main() -> None:
    
    lr_rate = 0.001
    record = []
    set_thres=0.19
    METRICS = [
            metrics.Precision(thresholds=set_thres),
            metrics.Recall(thresholds=set_thres),
            metrics.AUC()
    ]
    print(f'------------------------{cid}-----------------')
    
    # Load local data partition
    x_train, x_test, y_train, y_test = load_data(args.hospital)

    # Load and compile Keras model
    opt_adam = Adam(learning_rate=lr_rate)
    model = Sequential() 
    model.add(Dense(32, activation='relu', input_shape=(x_train.shape[1],))) #,kernel_regularizer='l2'
    model.add(Dense(16, activation='relu'))
    model.add(Dense(10, activation='relu'))    
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=opt_adam, loss=tf.losses.BinaryFocalCrossentropy(gamma=2.0), metrics=METRICS)

    # Start Flower client
    client = CifarClient(model, x_train, y_train, x_test, y_test, cid, record)
    fl.client.start_numpy_client("[::]:5656", client=client)

if __name__ == "__main__":
    main()
