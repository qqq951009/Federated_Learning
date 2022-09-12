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
import utils


# Parse command line argument `partition`
parser = argparse.ArgumentParser(description="Flower")
parser.add_argument("--seed", type=int, choices=range(0, 1000), required=True)
parser.add_argument("--seer", type=int, required=True)
args = parser.parse_args()

#SEED
size=0.2
seed = args.seed
seer = args.seer
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

imputation = utils.imputation()
target_encoding = utils.target_encoding(False)
split = utils.split_data(size, seed)

if seer == 1:
  columns = ["Class","LOC", "Gender", "Age", "AJCCstage", "DIFF", "LYMND", "TMRSZ", "SSF1", "SSF2"]
  df = pd.read_csv(r'/home/refu0917/lungcancer/server/AllCaseCtrl_final.csv')
  dfseer = pd.read_csv(r'/home/refu0917/lungcancer/data/seerdb.csv',index_col = [0])
  df["Class"] = df['Class'].apply(lambda x:1 if x != 0 else 0)
  df = df[columns]
  dfseer = dfseer[columns]
  dfimp = df.fillna(9)
  df_all = pd.concat([dfimp, dfseer])
  dfenc = target_encoding(df_all)
 

  output_file_name = 'centralized_score_seer.csv'
elif seer == 0:
  columns = ["Class","LOC", "FullDate","Gender", "Age", "CIG",
            "ALC", "BN", "MAGN", "AJCCstage", "DIFF", "LYMND",
            "TMRSZ", "OP", "RTDATE", "STDATE", "BMI_label",
            "SSF1", "SSF2", "SSF3", "SSF4", "SSF6"]
  df = pd.read_csv(r'/home/refu0917/lungcancer/server/AllCaseCtrl_final.csv')
  df = df[columns]
  df["Class"] = df['Class'].apply(lambda x:1 if x != 0 else 0)
  dfimp = imputation(df ,'drop_and_fill')
  dfenc = target_encoding(dfimp)
  output_file_name = 'centralized_score.csv'


def main() -> None:
    
    lr_rate = 0.001
    epoch=100
    set_thres=0.19
    METRICS = [
            metrics.Precision(thresholds=set_thres),
            metrics.Recall(thresholds=set_thres),
            metrics.AUC()
    ]
    
    # Load train data
    x_train, y_train, testset_dict = split(dfenc)
   

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
    score_df = pd.read_csv(output_file_name,index_col=[0])
    for i in sorted(dfenc['LOC'].unique()):
      df_test = testset_dict[i]
      x_test = df_test.drop(columns = ['Class'])
      y_test = df_test['Class']
      y_pred = model.predict(x_test)
      temp_auc_score.append(roc_auc_score(y_test, y_pred))
    score_df.loc[seed] = temp_auc_score
    score_df.to_csv(output_file_name)

if __name__ == "__main__":
    main()