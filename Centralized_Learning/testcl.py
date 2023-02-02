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
import mlflow.tensorflow

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# physical_devices = tf.config.list_physical_devices("CPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Parse command line argument `partition`
parser = argparse.ArgumentParser(description="Flower")
parser.add_argument("--seed", type=int, choices=range(0, 1000), required=True)
parser.add_argument("--seer", type=int, default=0)
args = parser.parse_args()

#SEED
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
  output_file_name = 'centralized_score_(1-3,0).csv'
  columns = ["Class","LOC", "Gender", "Age", "CIG",
            "ALC", "BN", "MAGN", "AJCCstage", "DIFF", "LYMND",
            "TMRSZ", "OP", "RTDATE", "STDATE", "BMI_label",
            "SSF1", "SSF2", "SSF3", "SSF4", "SSF6"] # "FullDate",
  df = pd.read_csv(config['data_dir']['8hos'],index_col=[0])
  df = df[columns]
  site_list = [2,3,6,8,9,10,11,12]

trainset, testset = preprocess_df(df, site_list)

# Impute the trainset and testset respectively
trainimp, testimp = imputation_fn(trainset, testset, config['imp_method'], seed)
# Encode trainset and map the encode dictionary to testset
x_train, y_train, testenc  = onehot_encode(trainimp, testimp)
# x_train, y_train, testenc  = target_encode(trainimp, testimp)
x_test_enc, y_test_enc = testenc.drop(columns=['Class','LOC']),testenc['Class']

def main() -> None:
    METRICS = [
            metrics.Precision(thresholds=set_thres),
            metrics.Recall(thresholds=set_thres),
            metrics.AUC()
    ]

    # Load and compile Keras model
    opt_adam = Adam(learning_rate=0.001)# , decay=0.005
    model = Sequential() 
    model.add(Dense(32, activation='relu', input_shape=(x_train.shape[1],))) #,kernel_regularizer='l2'
    model.add(Dense(16, activation='relu'))
    model.add(Dense(10, activation='relu'))    
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=opt_adam, loss=tf.losses.BinaryFocalCrossentropy(gamma=2.0), metrics=METRICS)

    model.fit(x_train,y_train,batch_size=16,epochs=100,verbose=2, validation_data=(x_test_enc, y_test_enc))

    temp_auc_score = []
    score_df = pd.read_csv(dir_name+output_file_name,index_col=[0])
    for i in sorted(df['LOC'].unique()):
      df_test = testenc.loc[testenc['LOC'] == i]
      x_test = df_test.drop(columns = ['Class','LOC'])
      y_test = df_test['Class']
      y_pred = model.predict(x_test)
      temp_auc_score.append(roc_auc_score(y_test, y_pred))
    print(lr_rate, decay)
    print(temp_auc_score)
    score_df.loc[seed] = temp_auc_score
    score_df.to_csv(dir_name+output_file_name)

if __name__ == "__main__":
    main()
