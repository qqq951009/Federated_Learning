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
# seed = 48,60,65,101
dir_name = '/home/refu0917/lungcancer/remote_output1/output_folder/cl_folder/'
output_file_name = 'centralized_score.csv'
df = pd.read_csv(dir_name+output_file_name)
print(df.head())

'''#SEED
size=0.2
seed = 60
seer = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

map = utils.mapping()
drop_year = utils.drop_year()
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

preprocess_df = utils.preprocess(size, seed)
trainset, testset = preprocess_df(df, site_list)
temp = testset[testset['LOC'] == 8] 
print(temp.Class.value_counts())'''