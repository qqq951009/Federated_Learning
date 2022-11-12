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

parser = argparse.ArgumentParser(description="Flower")
parser.add_argument("--hospital", type=int, choices=range(0, 10), required=True)
parser.add_argument("--seed", type=int, choices=range(0, 1000), required=True)
parser.add_argument("--seer", type=int, required=True)
# parser.add_argument("--encode_dict", type=str, required=True)   # put weight or average
args = parser.parse_args()

#SEED

lr_rate = 0.001
epoch = 100
size=0.2
auc_val_result = {}
hospital_list = [2,3,6,8]

set_thres=0.19
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
dir_name = '/home/refu0917/lungcancer/remote_output1/output_folder/fill_median_folder_check/'
map = utils.mapping()
drop = utils.drop_year_and_null()
imputation_fn = utils.imputation()
iterative_imputation = utils.iterative_imputation()
target_encode = utils.target_encoding()
train_enc_map_fn = utils.train_enc_map()

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
    output_file_name = 'local_val_df.csv'
    columns = ["Class","LOC", "FullDate","Gender", "Age", "CIG",
            "ALC", "BN", "MAGN", "AJCCstage", "DIFF", "LYMND",
            "TMRSZ", "OP", "RTDATE", "STDATE", "BMI_label",
            "SSF1", "SSF2", "SSF3", "SSF4", "SSF6"]
    df = pd.read_csv(r'/home/refu0917/lungcancer/server/AllCaseCtrl_final.csv')
    df = df[columns]

df['Class'] = df['Class'].apply(lambda x:1 if x != 0 else 0)
df = df[df['LOC'] == site_id]


df = drop(df)
# Split df into train and test set
trainset, testset = train_test_split(df,test_size = size,stratify=df['Class'],random_state=seed)
# Impute the trainset and testset respectively
trainimp, testimp = imputation_fn(trainset, testset, 'median')

# Encode trainset and map the encode dictionary to testset
x_train, y_train, x_test, y_test = target_encode(trainimp, testimp)

opt_adam = Adam(learning_rate=lr_rate)
model = Sequential() 
model.add(Dense(32, activation='relu', input_shape=(x_train.shape[1],))) #,kernel_regularizer='l2'
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='relu'))    
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=opt_adam, loss=tf.losses.BinaryFocalCrossentropy(gamma=2.0), metrics=METRICS)


hist = model.fit(x_train,y_train,batch_size=16,epochs=epoch,verbose=2,validation_data=(x_test, y_test))
y_pred = model.predict(x_test)
auc_val_result[str(site_id)] = [roc_auc_score(y_test, y_pred)]
val_df = pd.read_csv(dir_name+output_file_name, index_col=[0])
val_df.loc[seed,f'site{site_id}'] = roc_auc_score(y_test, y_pred)
val_df.to_csv(dir_name+output_file_name)
print(f'AUC by sklearn : {roc_auc_score(y_test,y_pred)}')