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
parser.add_argument("--seed", type=int, choices=range(0, 1000), required=True)
#parser.add_argument("--hospital", type=int, choices=range(0, 1000), required=True)
args = parser.parse_args()

#SEED
seed = args.seed
#hospital = args.hospital
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

def split_test(testset_dict, id):
    testset = testset_dict[id]
    x_test = testset.drop(columns = ['Class'])
    y_test = testset['Class']
    return x_test, y_test

def split_data(df,testsize,seed):
    testset_dict = {}
    df2 = df[df['LOC'] == 2]
    df3 = df[df['LOC'] == 3]
    df6 = df[df['LOC'] == 6]
    df8 = df[df['LOC'] == 8]

    df2 = df2.drop(columns=['LOC'])
    df3 = df3.drop(columns=['LOC'])
    df6 = df6.drop(columns=['LOC'])
    df8 = df8.drop(columns=['LOC'])

    trainset2,testset2 = train_test_split(df2,test_size = testsize,stratify=df2['Class'],random_state=seed)
    trainset3,testset3 = train_test_split(df3,test_size = testsize,stratify=df3['Class'],random_state=seed)
    trainset6,testset6 = train_test_split(df6,test_size = testsize,stratify=df6['Class'],random_state=seed)
    trainset8,testset8 = train_test_split(df8,test_size = testsize,stratify=df8['Class'],random_state=seed)
    trainset = pd.concat([trainset2,trainset3,trainset6,trainset8])
    trainset = shuffle(trainset)
    x_train = trainset.drop(columns=['Class'])
    y_train = trainset['Class']

    for i, j in zip([2,3,6,8], [testset2, testset3, testset6, testset8]) :
      testset_dict[i] = j

    return x_train, y_train, testset_dict

def load_data(train,test):
    df = pd.read_csv(r'/home/refu0917/lungcancer/server/AllCaseCtrl_final.csv')
    df = df[["Class","LOC", "FullDate","Gender", "Age", "CIG", "ALC", "BN",    #"FullDate",
            "MAGN", "AJCCstage", "DIFF", "LYMND", "TMRSZ",
            "OP", "RTDATE", "STDATE", "BMI_label",
            "SSF1", "SSF2", "SSF3", "SSF4", "SSF6"]]
    df['Class'] = df['Class'].apply(lambda x:1 if x != 0 else 0)

    # Ddata preprocess 
    df = drop_and_fill(df)
    df = target_encoding_loo(df)
    x, y, testset_dict = split_data(df,0.2,seed)
    return x, y

def main() -> None:
    
    lr_rate = 0.001
    epoch=100
    record = []
    set_thres=0.19
    METRICS = [
            metrics.Precision(thresholds=set_thres),
            metrics.Recall(thresholds=set_thres),
            metrics.AUC()
    ]
    
    # Load train data
    x_train, y_train, testset_dict = load_data()

    # Load and compile Keras model
    opt_adam = Adam(learning_rate=lr_rate)
    model = Sequential() 
    model.add(Dense(32, activation='relu', input_shape=(x_train.shape[1],))) #,kernel_regularizer='l2'
    model.add(Dense(16, activation='relu'))
    model.add(Dense(10, activation='relu'))    
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=opt_adam, loss=tf.losses.BinaryFocalCrossentropy(gamma=2.0), metrics=METRICS)

    model.fit(x_train,y_train,batch_size=16,epochs=epoch,verbose=2)

    temp_auc_score = []
    score_df = pd.read_csv('centralized_score.csv',index_col=[0])
    for i in [2,3,6,8]:
      x_test, y_test = split_test(testset_dict,i)
      y_pred = model.predict(x_test)
      temp_auc_score.append(roc_auc_score(y_test, y_pred))
    score_df.loc[seed] = temp_auc_score
    score_df.to_csv('centralized_score.csv')


if __name__ == "__main__":
    main()