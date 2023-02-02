import os 
import yaml
import random
import pickle
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras import metrics
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_auc_score
import utils
import mlflow.tensorflow

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
parser = argparse.ArgumentParser(description="Flower")
parser.add_argument("--hospital", type=int, choices=range(0, 100), required=True)
parser.add_argument("--seed", type=int, default=42, choices=range(0, 1000))
parser.add_argument("--seer", type=int, default=0)
# parser.add_argument("--encode_dict", type=str, required=True)   # put weight or average
args = parser.parse_args()

#SEED
with open('../config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.Loader)

epoch = config['epoch']
lr_rate = config['lr_rate']
size = config['test_size']
dir_name = config['dir_name']
set_thres = config['set_thres']

auc_val_result = {}
# hospital_list = [2,3,6,8]


METRICS = [
      metrics.Precision(thresholds=set_thres),
      metrics.Recall(thresholds=set_thres),
      metrics.AUC()
]

site_id = args.hospital
seed = args.seed
seer = args.seer

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

drop = utils.drop_year_and_null()
imputation_fn = utils.imputation()
target_encode = utils.target_encoding()
onehot_encode = utils.onehot_encoding()

if seer == 1:
    output_file_name = 'local_val_df_seer.csv'
    columns = ["Class","LOC", "FullDate", "Gender", "Age", "AJCCstage", 
                "DIFF", "LYMND", "TMRSZ", "SSF1", "SSF2", "SSF4", "OP"]
    df1 = pd.read_csv(r'/home/refu0917/lungcancer/server/AllCaseCtrl_final.csv')
    df2 = pd.read_csv(r'/home/refu0917/lungcancer/data/seerdb.csv',index_col = [0])
    df1["Class"], df2["Class"] = df1['Class'].apply(lambda x:1 if x != 0 else 0), df2['Class'].apply(lambda x:2 if x != 0 else 1)
    df1, df2 = df1[columns], df2[columns]
    df = pd.concat([df1, df2])

elif seer == 0:
    output_file_name = 'local_val_df_wdecay_0.0001_150epoch.csv'
    columns = ["Class","LOC", "Gender", "Age", "CIG",
            "ALC", "BN", "MAGN", "AJCCstage", "DIFF", "LYMND",
            "TMRSZ", "OP", "RTDATE", "STDATE", "BMI_label",
            "SSF1", "SSF2", "SSF3", "SSF4", "SSF6"] # "FullDate",
    df = pd.read_csv(config['data_dir']['8hos'],index_col=[0])
    df = df[columns]

df = df[df['LOC'] == site_id]

# Split df into train and test set
trainset, testset = train_test_split(df, test_size = size, stratify = df['Class'], random_state = seed)

# Impute the trainset and testset respectively
trainimp, testimp = imputation_fn(trainset, testset, config['imp_method'], seed)

# Encode trainset and map the encode dictionary to testset
x_train, y_train, x_test, y_test = target_encode(trainimp, testimp)
# x_train, y_train, x_test, y_test = onehot_encode(trainimp, testimp)

def main() -> None:
    #mlflow.tensorflow.autolog()
    #mlflow.set_experiment("Localized (Target)")
    #mlflow.set_tag("mlflow.runName", "site"+str(site_id)+'_'+str(seed)+'_'+str(lr_rate))

    # callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
    opt_adam = Adam(learning_rate=lr_rate)  # 
    model = Sequential() 
    model.add(Dense(32, activation='relu', input_shape=(x_train.shape[1],))) #,kernel_regularizer='l2'
    model.add(Dense(16, activation='relu'))
    model.add(Dense(10, activation='relu'))    
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=opt_adam, loss=tf.losses.BinaryFocalCrossentropy(gamma=2.0), metrics=METRICS)


    hist = model.fit(x_train,y_train,batch_size=16,epochs=epoch,verbose=2,validation_data=(x_test, y_test))
    y_pred = model.predict(x_test)
    auc_val_result[str(site_id)] = [roc_auc_score(y_test, y_pred)]
    # val_df = pd.read_csv(dir_name + output_file_name, index_col=[0])
    # val_df.loc[seed,f'site{site_id}'] = roc_auc_score(y_test, y_pred)
    # val_df.to_csv(dir_name + output_file_name)
    print(f'AUC by sklearn : {roc_auc_score(y_test,y_pred)}')

if __name__ == "__main__":
    main()
