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
import utils



#SEED
seed = 42
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

def drop_and_fill(df):
  df['year'] = [int(x[:4]) for x in list(df['FullDate'])]
  df['null_count'] = list(df.isna().sum(axis=1))
  df = df.reset_index(drop = True)
  index = [i.tolist() for i in np.where( (df['null_count']  >= 9) & (df['year'] <= 2010))][0]
  df = df.iloc[~df.index.isin(index)]
  #df = df.fillna(df.median())
  #df = df.drop(columns = ['year','null_count','FullDate'])
  #df = df.astype(int)
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

drop_year_fn = utils.drop_year()

df = pd.read_csv(r'/home/refu0917/lungcancer/server/AllCaseCtrl_final.csv')
df = df[["Class","LOC", "FullDate","Gender", "Age", "CIG", "ALC", "BN",    #"FullDate",
        "MAGN", "AJCCstage", "DIFF", "LYMND", "TMRSZ",
        "OP", "RTDATE", "STDATE", "BMI_label",
        "SSF1", "SSF2", "SSF3", "SSF4", "SSF6"]]
df['Class'] = df['Class'].apply(lambda x:1 if x != 0 else 0)
df = df[df['LOC'] == 3]
df = drop_year_fn(df)


trainset, testset = train_test_split(df,test_size = 0.2,stratify=df['Class'],random_state=seed)
imputation_fn = utils.imputation()
trainimp, testimp = imputation_fn(trainset, testset, 'drop_and_fill')
dfimp = pd.concat([trainimp,testimp])
print(len(dfimp))

