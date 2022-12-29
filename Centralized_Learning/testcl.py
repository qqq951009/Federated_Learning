import os 
import yaml
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
parser.add_argument("--seed", type=int, default=42, choices=range(0, 1000), required=True)
parser.add_argument("--seer", type=int, default=0)
args = parser.parse_args()

#SEED
with open('../config.yaml', 'r') as f:
      config = yaml.load(f, Loader=yaml.Loader)

epoch = config['epoch']
lr_rate = config['lr_rate']
size = config['test_size']
dir_name = config['dir_name']
set_thres = config['set_thres']

seed = args.seed
seer = args.seer
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

drop = utils.drop_year_and_null()
preprocess_df = utils.preprocess(size, seed)
imputation_fn = utils.imputation()
target_encode = utils.target_encoding()
onehot_encode = utils.onehot_encoding()

if seer == 1:
  output_file_name = 'centralized_score_seer.csv'
  columns = ["Class","LOC", "Gender", "Age", "AJCCstage", 
             "DIFF", "LYMND", "TMRSZ", "SSF1", "SSF2", "SSF4", "OP"] # "FullDate", 
  df1 = pd.read_csv(r'/home/refu0917/lungcancer/server/AllCaseCtrl_final.csv')
  df2 = pd.read_csv(r'/home/refu0917/lungcancer/data/seerdb.csv',index_col = [0])
  df1["Class"], df2["Class"] = df1['Class'].apply(lambda x:1 if x != 0 else 0), df2['Class'].apply(lambda x:1 if x != 0 else 0)
  df1, df2 = df1[columns], df2[columns]
  df = pd.concat([df1, df2])
  site_list = [2,3,6,8,9]
  
elif seer == 0:
  output_file_name = 'centralized_score.csv'
  columns = ["Class","LOC", "Gender", "Age", "CIG",
            "ALC", "BN", "MAGN", "AJCCstage", "DIFF", "LYMND",
            "TMRSZ", "OP", "RTDATE", "STDATE", "BMI_label",
            "SSF1", "SSF2", "SSF3", "SSF4", "SSF6"] # "FullDate",
  df = pd.read_csv(config['data_dir']['8hos'],index_col=[0])
  df = df[columns]
  site_list = [2,3,6,8,9,10,11,12]

# Drop the year smaller than 2010
trainset, testset = preprocess_df(df, site_list)

# Impute the trainset and testset respectively
trainimp, testimp = imputation_fn(trainset, testset, config['imp_method'], seed)
x_train, y_train, testset = onehot_encode(trainimp, testimp)
'''x_train, y_train = trainimp.drop(columns=['Class', 'LOC']), trainimp['Class']
x_test, y_test = testimp.drop(columns=['Class', 'LOC']), testimp['Class']

x_train_onehot = pd.get_dummies(x_train.astype(str))
test_onehot = pd.get_dummies(x_test.astype(str))
test_onehot['Class'] = y_test
test_onehot['LOC'] = testimp['LOC']'''

def main() -> None:

    METRICS = [
            metrics.Precision(thresholds=set_thres),
            metrics.Recall(thresholds=set_thres),
            metrics.AUC()
    ]
    print(x_train.shape[1])
    # Load and compile Keras model
    opt_adam = Adam(learning_rate=0.01)
    model = Sequential() 
    model.add(Dense(64, activation='relu', input_shape=(x_train.shape[1],))) #,kernel_regularizer='l2'
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(10, activation='relu'))    
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=opt_adam, loss=tf.losses.BinaryFocalCrossentropy(gamma=2.0), metrics=METRICS)
    model.fit(x_train,y_train,batch_size=64,epochs=epoch,verbose=2, validation_split=0.2)

    temp_auc_score = []
    for i in sorted(df['LOC'].unique()):
      df_test = x_train.loc[x_train['LOC'] == i]
      x_test = df_test.drop(columns = ['Class','LOC'])
      y_test = df_test['Class']
      y_pred = model.predict(x_test)
      temp_auc_score.append(roc_auc_score(y_test, y_pred))
    print(temp_auc_score)
    

if __name__ == "__main__":
    main()
