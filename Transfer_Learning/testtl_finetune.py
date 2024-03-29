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
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_auc_score
import utils
import mlflow.tensorflow

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Parse command line argument `partition`
parser = argparse.ArgumentParser(description="Flower")
parser.add_argument("--hospital", type=int, choices=range(0, 1000), required=True)
parser.add_argument("--seed", type=int, default=42, choices=range(0, 1000))
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

METRICS = [
      metrics.Precision(thresholds=set_thres),
      metrics.Recall(thresholds=set_thres),
      metrics.AUC()
]

seed = args.seed
seer = args.seer
hospital = args.hospital
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
imputation_fn = utils.imputation()
onehot_encode = utils.onehot_encoding()

index_list, df_list = [], []

if seer == 1:
  columns = ["Class","LOC", "FullDate", "Gender", "Age", "AJCCstage", 
             "DIFF", "LYMND", "TMRSZ", "SSF1", "SSF2", "SSF4", "OP"]
  df1 = pd.read_csv(r'/home/refu0917/lungcancer/server/AllCaseCtrl_final.csv')
  df2 = pd.read_csv(r'/home/refu0917/lungcancer/data/seerdb.csv',index_col = [0])
  df1, df2 = df1[columns], df2[columns]
  df = pd.concat([df1, df2], axis = 0)
  df['Class'] = df['Class'].apply(lambda x:1 if x != 0 else 0)
  output_file_name = 'transfer_learning_score_seer.csv'

elif seer == 0:
  columns = ["Class","LOC", "Gender", "Age", "CIG",
            "ALC", "BN", "MAGN", "AJCCstage", "DIFF", "LYMND",
            "TMRSZ", "OP", "RTDATE", "STDATE", "BMI_label",
            "SSF1", "SSF2", "SSF3", "SSF4", "SSF6"] # "FullDate",
  df = pd.read_csv(config['data_dir']['8hos'],index_col=[0])
  df = df[columns]
  output_file_name = f'transfer_score_({lr_rate},{decay}).csv'


dfencode = pd.DataFrame()
train_index, test_index = [], []
for i in [2,3,6,8,9,10,11,12]:
    tempdf = df[df['LOC'] == i]
    train, test = train_test_split(tempdf, test_size = size, stratify = tempdf['Class'], random_state = seed)
    if i == hospital:
        train_index = train.index
        test_index = test.index
    trainimp, testimp = imputation_fn(train, test, config['imp_method'], seed)
    dfencode = pd.concat([dfencode, trainimp, testimp])


x_train, y_train, x_test, y_test = onehot_encode(dfencode, hospital, train_index, test_index)


def main() -> None:
    mlflow.tensorflow.autolog()
    mlflow.set_experiment("Transfer (Onehot_new)")
    mlflow.set_tag("mlflow.runName", "site"+str(hospital)+'_'+str(seed)+'_('+str(lr_rate)+','+str(decay)+')')

    # Load and compile Keras model
    opt_adam = Adam(learning_rate=lr_rate, decay=decay)
    model = keras.models.load_model('pretrained_model')
    model.compile(optimizer=opt_adam, loss=tf.losses.BinaryFocalCrossentropy(gamma=2.0), metrics=METRICS)
    model.fit(x_train, y_train, batch_size = 16, epochs = epoch, verbose=2, validation_data=(x_test, y_test))
    y_pred = model.predict(x_test)

    print(f'AUC by sklearn : {roc_auc_score(y_test,y_pred)}')
    score_df = pd.read_csv(dir_name + output_file_name,index_col=[0])
    score_df.loc[seed,f"site{hospital}"] = roc_auc_score(y_test, y_pred)
    score_df.to_csv(dir_name + output_file_name)

if __name__ == "__main__":
    main()