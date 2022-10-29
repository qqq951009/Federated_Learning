import os
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

# Parse command line argument `partition`
parser = argparse.ArgumentParser(description="Flower")
parser.add_argument("--seed", type=int, choices=range(0, 1000), required=True)
parser.add_argument("--seer", type=int, required=True)
args = parser.parse_args()


size = 0.2
seed = args.seed
seer = args.seer
dir_name = '/home/refu0917/lungcancer/remote_output1/output_folder/imputation_test_folder/'
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
lr_rate = 0.001
epoch = 100
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
  columns = ["Class","LOC", "FullDate","Gender", "Age", "CIG",
            "ALC", "BN", "MAGN", "AJCCstage", "DIFF", "LYMND",
            "TMRSZ", "OP", "RTDATE", "STDATE", "BMI_label",
            "SSF1", "SSF2", "SSF3", "SSF4", "SSF6"]
  df = pd.read_csv(r'/home/refu0917/lungcancer/server/AllCaseCtrl_final.csv')
  df = df[columns]
  df['Class'] = df['Class'].apply(lambda x:1 if x != 0 else 0)
  output_file_name = 'transfer_learning_score.csv'

for i in [2,3,6,8]:
    tempdf = df[df['LOC'] == i]
    df_list += [tempdf]
    index_list.append(i)

def main() -> None:
    lr_rate = 0.001
    epoch=100
    set_thres=0.19
    METRICS = [
            metrics.Precision(thresholds=set_thres),
            metrics.Recall(thresholds=set_thres),
            metrics.AUC()
    ]
    

    with open('imputationdf.pickle', 'rb') as f:
        site_imp_dict = pickle.load(f)
    with open('mapping.pickle', 'rb') as f:
        site_map_dict = pickle.load(f)

    # utils.pretrain_site 找出最資料量最大的醫院當作pretrain model
    ptr_site = utils.pretrain_site()
    map = utils.mapping()

    # Choose the biggest site among all sites
    ptr_siteID = ptr_site(df_list, index_list)

    # Select the biggest site from the dataframe after imputation
    dfimp = site_imp_dict[ptr_siteID]
    trainimp, testimp = dfimp['train'],dfimp['test']
    
    # Map the target encoding
    trainenc = map(site_map_dict, trainimp, columns[3:])
    testenc = map(site_map_dict, testimp, columns[3:])
    trainenc['Class'] = trainenc['Class'].apply(lambda x:1 if x!=1 else 0)
    testenc['Class'] = testenc['Class'].apply(lambda x:1 if x!=1 else 0)
    
    # Split X and Y
    x_train,y_train = trainenc.drop(columns = ['Class', 'LOC']), trainenc['Class']
    x_test, y_test = testenc.drop(columns = ['Class', 'LOC']), testenc['Class']

   

    # Load and compile Keras model
    opt_adam = Adam(learning_rate=lr_rate)
    model = Sequential() 
    model.add(Dense(32, activation='relu', input_shape=(x_train.shape[1],))) #,kernel_regularizer='l2'
    model.add(Dense(16, activation='relu'))
    model.add(Dense(10, activation='relu'))    
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=opt_adam, loss=tf.losses.BinaryFocalCrossentropy(gamma=2.0), metrics=METRICS)

    model.fit(x_train,y_train,batch_size=16,epochs=epoch,verbose=2,validation_data=(x_test, y_test))
    model.save('pretrained_model')

    y_pred = model.predict(x_test)
    score_df = pd.read_csv(dir_name + output_file_name, index_col=[0])
    score_df.loc[seed,f"site{ptr_siteID}"] = roc_auc_score(y_test, y_pred)
    score_df.to_csv(dir_name + output_file_name)

if __name__ == "__main__":
    main()
