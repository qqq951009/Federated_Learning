import os
import keras
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
parser.add_argument("--seed", type=int, choices=range(0, 1000), required=True)
parser.add_argument("--hospital", type=int, choices=range(0, 1000), required=True)
args = parser.parse_args()

seed = args.seed
hospital = args.hospital
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

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
  encoder = TargetEncoder(cols=columns,smoothing=0.05)
  df_target = encoder.fit_transform(x,y)
  df_target['Class'] = y
  return df_target

def split_data(df,testsize,seed):
    df = df.drop(columns=['LOC'])
    trainset,testset = train_test_split(df,test_size = testsize,stratify=df['Class'],random_state=seed)
    x_train = trainset.drop(columns=['Class'])
    x_test = testset.drop(columns=['Class'])
    y_train = trainset['Class']
    y_test = testset['Class']
    return x_train,x_test,y_train ,y_test 

def load_data():
    df = pd.read_csv(r'/home/refu0917/lungcancer/server/AllCaseCtrl_final.csv')
    df = df[["Class","LOC", "FullDate","Gender", "Age",  
            "AJCCstage", "DIFF", "LYMND", "TMRSZ",
            "SSF1", "SSF2"]]

    df['Class'] = df['Class'].apply(lambda x:1 if x != 0 else 0)
    df = df[df['LOC'] == hospital]

    # Ddata preprocess
    df = drop_and_fill(df)
    df = target_encoding_loo(df)
    x_train,x_test,y_train,y_test = split_data(df,0.2,seed)
    return x_train,x_test,y_train,y_test

def main() -> None:
    
    lr_rate = 0.001
    record = []
    set_thres=0.19
    METRICS = [
            metrics.Precision(thresholds=set_thres),
            metrics.Recall(thresholds=set_thres),
            metrics.AUC()
    ]
    
    # Load local data partition
    x_train, x_test, y_train, y_test = load_data()

    # Load and compile Keras model
    model = keras.models.load_model('pretrained_model')
    hist = model.fit(x_train,y_train,batch_size=16,epochs=100,verbose=2,validation_data=(x_test, y_test))
    y_pred = model.predict(x_test)
    score_df = pd.read_csv('transfer_learning_score.csv',index_col=[0])
    score_df.loc[seed,f"site{hospital}"] = roc_auc_score(y_test, y_pred)
    score_df.to_csv('transfer_learning_score.csv')

if __name__ == "__main__":
    main()