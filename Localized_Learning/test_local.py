import random
import pickle
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras import metrics
import matplotlib.pyplot as plt
import scikitplot as skplt
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
dir_name = '/home/refu0917/lungcancer/remote_output1/output_folder/drop_and_fill_folder/'
map = utils.mapping()
drop_year = utils.drop_year()
drop_year_null = utils.drop_year_null()
iterative_imputation = utils.iterative_imputation()
target_encode = utils.target_encoding(False)
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
    output_file_name = 'local_val_df_comp.csv'
    columns = ["Class","LOC", "FullDate","Gender", "Age", "CIG",
            "ALC", "BN", "MAGN", "AJCCstage", "DIFF", "LYMND",
            "TMRSZ", "OP", "RTDATE", "STDATE", "BMI_label",
            "SSF1", "SSF2", "SSF3", "SSF4", "SSF6"]
    df = pd.read_csv(r'/home/refu0917/lungcancer/server/AllCaseCtrl_final.csv')
    df = df[columns]

df['Class'] = df['Class'].apply(lambda x:1 if x != 0 else 0)
df = df[df['LOC'] == site_id]

imputation_fn1 = utils.imputation()
imputation_fn2 = utils.imputation2()

# Drop the year smaller than 2010
df1 = drop_year_null(df)
df2 = drop_year(df)
print(df1.isna().sum(axis=1).value_counts())
print(df2.isna().sum(axis=1).value_counts())
print(len(df1), len(df2))
# Split df into train and test set
trainset1, testset1 = train_test_split(df1,test_size = size,stratify=df1['Class'],random_state=seed)
trainset2, testset2 = train_test_split(df2,test_size = size,stratify=df2['Class'],random_state=seed)
print(len(trainset1), len(testset1))
print(len(trainset2), len(testset2))
trainimp1, testimp1 = imputation_fn1(trainset1, testset1, 'drop_and_fill')
trainimp2, testimp2 = imputation_fn2(trainset2, testset2, 'drop_and_fill')
print(len(trainimp1), len(testimp1))
print(len(trainimp2), len(testimp2))