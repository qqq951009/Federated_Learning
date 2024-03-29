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
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten,Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_auc_score
import utils
import mlflow.tensorflow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Parse command line argument `partition`
parser = argparse.ArgumentParser(description="Flower")
parser.add_argument("--seed", type=int, choices=range(0, 1000), required=True)
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


seed = args.seed
seer = args.seer
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

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

for i in [2,3,6,8,9,10,11,12]:
    tempdf = df[df['LOC'] == i]
    df_list += [tempdf]
    index_list.append(i)

def main() -> None:
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
    
    mlflow.tensorflow.autolog()
    mlflow.set_experiment("Transfer (Target_new)")
    mlflow.set_tag("mlflow.runName", "site"+str(ptr_siteID)+'_'+str(seed)+'_('+str(lr_rate)+','+str(decay)+')')

    # Map the target encoding
    trainenc = map(site_map_dict, trainimp, columns[2:])
    testenc = map(site_map_dict, testimp, columns[2:])
    trainenc['Class'] = trainenc['Class'].apply(lambda x:1 if x!=1 else 0)
    testenc['Class'] = testenc['Class'].apply(lambda x:1 if x!=1 else 0)
    
    # Split X and Y
    x_train,y_train = trainenc.drop(columns = ['Class', 'LOC']), trainenc['Class']
    x_test, y_test = testenc.drop(columns = ['Class', 'LOC']), testenc['Class']



    # Load and compile Keras model
    opt_adam = Adam(learning_rate=lr_rate)  # , decay=decay
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