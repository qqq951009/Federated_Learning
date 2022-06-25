import random
import pickle
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import configparser
from keras import metrics
import matplotlib.pyplot as plt
import scikitplot as skplt
import tensorflow_addons as tfa
from sklearn.utils import shuffle
from imblearn.combine import SMOTETomek, SMOTEENN 
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam, Adagrad
from sklearn.metrics import roc_auc_score
from category_encoders import TargetEncoder, LeaveOneOutEncoder



parser = argparse.ArgumentParser(description="Flower")
parser.add_argument("--hospital", type=int, choices=range(0, 10), required=True)
parser.add_argument("--seed", type=int, choices=range(0, 1000), required=True)
args = parser.parse_args()

#SEED
seed = args.seed
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
lr_rate = 0.001
epoch = 100
auc_val_result = {}
hospital_list = [2,3,6,8]

set_thres=0.19
METRICS = [
      metrics.Precision(thresholds=set_thres),
      metrics.Recall(thresholds=set_thres),
      metrics.AUC()
]

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

# Load local data partition
x_train, x_test, y_train, y_test = load_data(args.hospital)

opt_adam = Adam(learning_rate=lr_rate)
model = Sequential() 
model.add(Dense(32, activation='relu', input_shape=(x_train.shape[1],))) #,kernel_regularizer='l2'
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='relu'))    
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=opt_adam, loss=tf.losses.BinaryFocalCrossentropy(gamma=2.0), metrics=METRICS)


hist = model.fit(x_train,y_train,batch_size=16,epochs=epoch,verbose=2,validation_data=(x_test, y_test))
y_pred = model.predict(x_test)
auc_val_result[str(args.hospital)] = [roc_auc_score(y_test, y_pred)]
val_df = pd.read_csv('./data/local_val_df.csv', index_col=[0])
val_df.loc[seed,f'site{args.hospital}'] = roc_auc_score(y_test, y_pred)
val_df.to_csv('local_val_df.csv')
print(f'AUC by sklearn : {roc_auc_score(y_test,y_pred)}')

# Use Local model to evaluate other hospital : 
hospital_list.remove(args.hospital)
for i in hospital_list:
  _,x_val,_,y_val = load_data(i)
  y_pred_val = model.predict(x_val)
  auc_val_result[str(i)] = [roc_auc_score(y_val, y_pred_val)]
auc_val_result = dict(sorted(auc_val_result.items()))  # Sort the AUC result dict
print(auc_val_result)

with open(f'./data/Local_AUC_val_{args.hospital}.pickle', 'wb') as f:
    pickle.dump(auc_val_result, f)

with open(f'./data/Local_AUC_{args.hospital}.pickle', 'wb') as f:
    pickle.dump(hist.history['val_auc'], f)

local_auc_df = pd.read_csv(f'./data/local_auc_df{args.hospital}.csv')
local_auc_df[f'seed{seed}'] = hist.history['val_auc']
local_auc_df.to_csv(f'./data/local_auc_df{args.hospital}.csv',index=False)
'''local_auc_df = pd.DataFrame(data = hist.history['val_auc'], columns=[f'seed{seed}'])
local_auc_df.to_csv(f'local_auc_df{args.hospital}.csv',index=False)'''
