import os
import yaml
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


parser = argparse.ArgumentParser(description="Flower")
parser.add_argument("--hospital", type=int, choices=range(0, 100), required=True)
parser.add_argument("--seed", type=int, choices=range(0, 1000), required=True)
parser.add_argument("--seer", type=int, default=0)
args = parser.parse_args()

with open('../config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.Loader)

epoch = config['epoch']
lr_rate = config['lr_rate']
size = config['test_size']
dir_name = config['dir_name']
set_thres = config['set_thres']

site_id = args.hospital
seed = args.seed
seer = args.seer

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

if seer == 1:
    columns = ["Class","LOC", "FullDate", "Gender", "Age", "AJCCstage", 
                "DIFF", "LYMND", "TMRSZ", "SSF1", "SSF2", "SSF4", "OP"]
      
elif seer == 0:
    columns = ["Class","LOC", "Gender", "Age", "CIG",
            "ALC", "BN", "MAGN", "AJCCstage", "DIFF", "LYMND",
            "TMRSZ", "OP", "RTDATE", "STDATE", "BMI_label",
            "SSF1", "SSF2", "SSF3", "SSF4", "SSF6"] # "FullDate",

with open('./encode_dict_folder/imputationdf.pickle', 'rb') as f:
    site_imp_dict = pickle.load(f)
with open('./encode_dict_folder/mapping.pickle', 'rb') as f:
    site_map_dict = pickle.load(f)
    


def main() -> None:
    
    METRICS = [
            metrics.Precision(thresholds=set_thres),
            metrics.Recall(thresholds=set_thres),
            metrics.AUC()
    ]
    print(f'------------------------{site_id}-----------------')
    
    map = utils.mapping()

    # Select the hospital from the dataframe after imputation
    dfimp = site_imp_dict[site_id]
    trainimp, testimp = dfimp['train'],dfimp['test']
    
    # Map the target encoding
    trainenc = map(site_map_dict, trainimp, columns[2:])
    testenc = map(site_map_dict, testimp, columns[2:])
    # Split X and Y
    x_train,y_train = trainenc.drop(columns = ['Class', 'LOC']), trainenc['Class']
    x_test, y_test = testenc.drop(columns = ['Class', 'LOC']), testenc['Class']

    # Load and compile Keras model
    opt_adam = Adam(learning_rate=lr_rate)
    model = Sequential() 
    model.add(Dense(32, activation='relu', input_shape=(x_train.shape[1],), name='base1')) #,kernel_regularizer='l2'
    model.add(Dense(16, activation='relu', name='base2'))
    model.add(Dense(10, activation='relu', name='base3'))    
    model.add(Dense(1, activation='sigmoid', name='personal'))
    # model.compile(optimizer=opt_adam, loss=tf.losses.BinaryFocalCrossentropy(gamma=2.0), metrics=METRICS)

    # Start Flower client
    client = utils.CifarClient_personal(model, x_train, y_train, x_test, y_test, site_id, size, seed, seer, config)
    fl.client.start_numpy_client("[::]:6000", client=client)

if __name__ == "__main__":
    main()
    