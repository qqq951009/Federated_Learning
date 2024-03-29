import os
import yaml
import keras
import pickle
import random
import argparse
import flwr as fl
import numpy as np
import pandas as pd
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
# Parse command line argument `partition`
parser = argparse.ArgumentParser(description="Flower")
parser.add_argument("--seed", type=int, choices=range(0, 1000), required=True)
parser.add_argument("--hospital", type=int, choices=range(0, 1000), required=True)
parser.add_argument("--seer", type=int, default=0)
args = parser.parse_args()

with open('../config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.Loader)

epoch = config['epoch']
lr_rate = config['lr_rate']
decay = config['decay']
size = config['test_size']
dir_name = config['dir_name']
set_thres = config['set_thres']


seed = args.seed
seer = args.seer
hospital = args.hospital
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

if seer == 1:
  columns = ["Class","LOC", "FullDate", "Gender", "Age", "AJCCstage", 
             "DIFF", "LYMND", "TMRSZ", "SSF1", "SSF2", "SSF4", "OP"]
  output_file_name = 'transfer_learning_score_seer.csv'

elif seer == 0:
  columns = ["Class","LOC", "Gender", "Age", "CIG",
            "ALC", "BN", "MAGN", "AJCCstage", "DIFF", "LYMND",
            "TMRSZ", "OP", "RTDATE", "STDATE", "BMI_label",
            "SSF1", "SSF2", "SSF3", "SSF4", "SSF6"] # "FullDate",
  output_file_name = f'transfer_score_({lr_rate},{decay}).csv'

with open('imputationdf.pickle', 'rb') as f:
    site_imp_dict = pickle.load(f)
with open('mapping.pickle', 'rb') as f:
    site_map_dict = pickle.load(f)

def main() -> None:
    mlflow.tensorflow.autolog()
    mlflow.set_experiment("Transfer (Target_new)")
    mlflow.set_tag("mlflow.runName", "site"+str(hospital)+'_'+str(seed)+'_('+str(lr_rate)+','+str(decay)+')')

    METRICS = [
            metrics.Precision(thresholds=set_thres),
            metrics.Recall(thresholds=set_thres),
            metrics.AUC()
    ]

    map = utils.mapping()

    # Select the hospital from the dataframe after imputation
    dfimp = site_imp_dict[hospital]
    trainimp, testimp = dfimp['train'],dfimp['test']
    
    # Map the target encoding
    trainenc = map(site_map_dict, trainimp, columns[2:])
    testenc = map(site_map_dict, testimp, columns[2:])
    trainenc['Class'] = trainenc['Class'].apply(lambda x:1 if x!=1 else 0)
    testenc['Class'] = testenc['Class'].apply(lambda x:1 if x!=1 else 0)
    
    # Split X and Y
    x_train,y_train = trainenc.drop(columns = ['Class', 'LOC']), trainenc['Class']
    x_test, y_test = testenc.drop(columns = ['Class', 'LOC']), testenc['Class']

    # Load and compile Keras model
    opt_adam = Adam(learning_rate=lr_rate)  # , decay=decay
    model = keras.models.load_model('pretrained_model')
    model.compile(optimizer=opt_adam, loss=tf.losses.BinaryFocalCrossentropy(gamma=2.0), metrics=METRICS)
    model.fit(x_train, y_train, batch_size = 16, epochs = epoch, verbose=2, validation_data=(x_test, y_test))
    y_pred = model.predict(x_test)
    score_df = pd.read_csv(dir_name + output_file_name,index_col=[0])
    score_df.loc[seed,f"site{hospital}"] = roc_auc_score(y_test, y_pred)
    score_df.to_csv(dir_name + output_file_name)

if __name__ == "__main__":
    main()