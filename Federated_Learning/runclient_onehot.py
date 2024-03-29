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
import mlflow.tensorflow

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
parser = argparse.ArgumentParser(description="Flower")
parser.add_argument("--hospital", type=int, choices=range(0, 100), required=True)
parser.add_argument("--seed", type=int, choices=range(0, 1000), required=True)

args = parser.parse_args()

with open('../config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.Loader)

epoch = config['epoch']
lr_rate = config['lr_rate']
decay = config['decay']
size = config['test_size']
dir_name = config['dir_name']
set_thres = config['set_thres']

site_id = args.hospital
seed = args.seed

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
METRICS = [
            metrics.Precision(thresholds=set_thres),
            metrics.Recall(thresholds=set_thres),
            metrics.AUC()
        ]
columns = ["Class","LOC", "Gender", "Age", "CIG",
        "ALC", "BN", "MAGN", "AJCCstage", "DIFF", "LYMND",
        "TMRSZ", "OP", "RTDATE", "STDATE", "BMI_label",
        "SSF1", "SSF2", "SSF3", "SSF4", "SSF6"]

df = pd.read_csv(config['data_dir']['8hos'],index_col=[0])
df = df[columns]
x_train, y_train, x_test, y_test = utils.onehot_aligment(df, seed, site_id, config)

def main() -> None:
    print(f'------------------------{site_id}-----------------')
    #mlflow.tensorflow.autolog()
    #mlflow.set_experiment("Federated (OneHot)")
    #mlflow.set_tag("mlflow.runName", "site"+str(site_id)+'_'+str(seed)+'_w-decay')

    # Load and compile Keras model
    # opt_adam = Adam(learning_rate=lr_rate, decay=decay)
    model = Sequential() 
    model.add(Dense(32, activation='relu', input_shape=(x_train.shape[1],), name='base1')) #,kernel_regularizer='l2'
    model.add(Dense(16, activation='relu', name='base2'))
    model.add(Dense(10, activation='relu', name='base3'))    
    model.add(Dense(1, activation='sigmoid', name='personal'))
    #model.compile(optimizer=opt_adam, loss=tf.losses.BinaryFocalCrossentropy(gamma=2.0), metrics=METRICS)

    # Start Flower client
    client = utils.CifarClient(model, x_train, y_train, x_test, y_test, site_id, size, seed, 0, config)
    fl.client.start_numpy_client("[::]:6000", client=client)

if __name__ == "__main__":
    main()