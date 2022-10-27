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
dir_name = '/home/refu0917/lungcancer/remote_output1/output_folder/cl_folder_test1/'
map = utils.mapping()
drop_year = utils.drop_year()
preprocess_df = utils.preprocess(size, seed)
iterative_imputation = utils.iterative_imputation()
target_encode = utils.target_encoding(False)
train_enc_map_fn = utils.train_enc_map()

if seer == 1:
  output_file_name = 'centralized_score_seer.csv'
  columns = ["Class","LOC", "FullDate", "Gender", "Age", "AJCCstage", 
             "DIFF", "LYMND", "TMRSZ", "SSF1", "SSF2", "SSF4", "OP"]
  df1 = pd.read_csv(r'/home/refu0917/lungcancer/server/AllCaseCtrl_final.csv')
  df2 = pd.read_csv(r'/home/refu0917/lungcancer/data/seerdb.csv',index_col = [0])
  df1["Class"], df2["Class"] = df1['Class'].apply(lambda x:1 if x != 0 else 0), df2['Class'].apply(lambda x:1 if x != 0 else 0)
  df1, df2 = df1[columns], df2[columns]
  df = pd.concat([df1, df2])
  site_list = [2,3,6,8,9]
  
elif seer == 0:
  output_file_name = 'centralized_score.csv'
  columns = ["Class","LOC", "FullDate","Gender", "Age", "CIG",
            "ALC", "BN", "MAGN", "AJCCstage", "DIFF", "LYMND",
            "TMRSZ", "OP", "RTDATE", "STDATE", "BMI_label",
            "SSF1", "SSF2", "SSF3", "SSF4", "SSF6"]
  df = pd.read_csv(r'/home/refu0917/lungcancer/server/AllCaseCtrl_final.csv')
  df["Class"] = df['Class'].apply(lambda x:1 if x != 0 else 0)
  df = df[columns]
  site_list = [2,3,6,8]

# Drop the year smaller than 2010
df = drop_year(df)
trainset, testset = preprocess_df(df, site_list)
#trainset, testset = train_test_split(df,test_size = size,stratify=df['Class'],random_state=seed)

# Impute the trainset and testset respectively
trainimp = iterative_imputation(trainset,seed)
testimp = iterative_imputation(testset,seed)

# Encode trainset and map the encode dictionary to testset
trainenc = target_encode(trainimp)
train_enc_dict = train_enc_map_fn(trainenc,trainimp, columns[3:],df)
testenc = map(train_enc_dict, testimp, columns[3:])

trainenc['Class'] = trainenc['Class'].apply(lambda x:1 if x!=1 else 0)
testenc['Class'] = testenc['Class'].apply(lambda x:1 if x!=1 else 0)

x_train,y_train = trainenc.drop(columns = ['Class', 'LOC']), trainenc['Class']
#x_test, y_test = testenc.drop(columns = ['Class', 'LOC']), testenc['Class']

def main() -> None:
    
    lr_rate = 0.001
    epoch=100
    set_thres=0.19
    METRICS = [
            metrics.Precision(thresholds=set_thres),
            metrics.Recall(thresholds=set_thres),
            metrics.AUC()
    ]

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
    score_df = pd.read_csv(dir_name+output_file_name,index_col=[0])
    for i in sorted(df['LOC'].unique()):
      df_test = testenc.loc[testenc['LOC'] == i]
      x_test = df_test.drop(columns = ['Class','LOC'])
      y_test = df_test['Class']
      y_pred = model.predict(x_test)
      temp_auc_score.append(roc_auc_score(y_test, y_pred))
    score_df.loc[seed] = temp_auc_score
    score_df.to_csv(dir_name+output_file_name)

if __name__ == "__main__":
    main()