import os
import time
import pickle
import random
import argparse
import flwr as fl
import numpy as np
import pandas as pd
import configparser
import tensorflow as tf
from keras import metrics
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten,Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_auc_score
import utils

config = configparser.ConfigParser()
config.read('config.ini')

parser = argparse.ArgumentParser(description="Flower")
parser.add_argument("--hospital", type=int, choices=range(0, 10), required=True)
parser.add_argument("--seed", type=int, choices=range(0, 1000), required=True)
parser.add_argument("--seer", type=int, required=True)
parser.add_argument("--encode_dict", type=str, required=True)   # put weight or average
args = parser.parse_args()

size = 0.2
site_id = args.hospital
seed = args.seed
seer = args.seer
average_weight = args.encode_dict

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

imputation = utils.imputation()
enc_dict = utils.choose_dict(average_weight,seer)
mapping = utils.mapping()
split = utils.split_data(size,seed)


if seer == 1:
    columns = ["Class","LOC", "Gender", "Age", "AJCCstage", "DIFF", "LYMND", "TMRSZ", "SSF1", "SSF2"]
    
    if site_id != 9:
        df = pd.read_csv(r'/home/refu0917/lungcancer/server/AllCaseCtrl_final.csv')
        df = df[columns]
        df['Class'] = df['Class'].apply(lambda x:1 if x != 0 else 0)
        df = df[df['LOC'] == site_id]
        dfimp = df.fillna(9)
        dfmap = mapping(enc_dict(),dfimp)

    elif site_id == 9:
        df = pd.read_csv(r'/home/refu0917/lungcancer/data/seerdb.csv',index_col = [0])
        df = df[columns]
        df = df[df['LOC'] == site_id]
        dfmap = mapping(enc_dict(),df)

        
elif seer == 0:
    columns = ["Class","LOC", "FullDate","Gender", "Age", "CIG",
            "ALC", "BN", "MAGN", "AJCCstage", "DIFF", "LYMND",
            "TMRSZ", "OP", "RTDATE", "STDATE", "BMI_label",
            "SSF1", "SSF2", "SSF3", "SSF4", "SSF6"]
    df = pd.read_csv(r'/home/refu0917/lungcancer/server/AllCaseCtrl_final.csv')
    df = df[columns]
    df["Class"] = df['Class'].apply(lambda x:1 if x != 0 else 0)
    df = df[df['LOC'] == site_id]
    dfimp = imputation(df, 'drop_and_fill')
    dfmap = mapping(enc_dict(),dfimp)



def main() -> None:
    
    lr_rate = 0.001
    record = []
    set_thres=0.19
    METRICS = [
            metrics.Precision(thresholds=set_thres),
            metrics.Recall(thresholds=set_thres),
            metrics.AUC()
    ]
    print(f'------------------------{site_id}-----------------')
    
    # Load local data partition
    x_train,x_test,y_train,y_test = split(dfmap)
    
    # Load and compile Keras model
    opt_adam = Adam(learning_rate=lr_rate)
    model = Sequential() 
    model.add(Dense(32, activation='relu', input_shape=(x_train.shape[1],))) #,kernel_regularizer='l2'
    model.add(Dense(16, activation='relu'))
    model.add(Dense(10, activation='relu'))    
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=opt_adam, loss=tf.losses.BinaryFocalCrossentropy(gamma=2.0), metrics=METRICS)

    # Start Flower client
    client = utils.CifarClient(model, x_train, y_train, x_test, y_test, site_id, size, seed, seer, average_weight)
    fl.client.start_numpy_client("[::]:7000", client=client)

if __name__ == "__main__":
    main()