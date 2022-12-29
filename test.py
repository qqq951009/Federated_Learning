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
from Localized_Learning import utils

parser = argparse.ArgumentParser(description="Flower")
parser.add_argument("--hospital", type=int, default=2)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--seer", type=int, default=0)
# parser.add_argument("--encode_dict", type=str, required=True)   # put weight or average
args = parser.parse_args()

#SEED
with open('config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.Loader)

epoch = config['epoch']
lr_rate = config['lr_rate']
size = config['test_size']
dir_name = config['dir_name']
set_thres = config['set_thres']

auc_val_result = {}
hospital_list = [2,3,6,8]


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

# map = utils.mapping()
drop = utils.drop_year_and_null()
imputation_fn = utils.imputation()

target_encode = utils.target_encoding()
#　train_enc_map_fn = utils.train_enc_map()

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
    columns = ["Class","LOC", "Gender", "Age", "CIG",
            "ALC", "BN", "MAGN", "AJCCstage", "DIFF", "LYMND",
            "TMRSZ", "OP", "RTDATE", "STDATE", "BMI_label",
            "SSF1", "SSF2", "SSF3", "SSF4", "SSF6"] # "FullDate",
    df = pd.read_csv(config['data_dir']['8hos'],index_col=[0])
    df = df[columns]

df = df[df['LOC'] == site_id]

# df = drop(df)
# Split df into train and test set
trainset, testset = train_test_split(df, test_size = size, stratify = df['Class'], random_state = seed)

# Impute the trainset and testset respectively
trainimp, testimp = imputation_fn(trainset, testset, config['imp_method'], seed)

# Encode trainset and map the encode dictionary to testset
x_train, y_train, x_test, y_test = target_encode(trainimp, testimp)

opt_adam = Adam(learning_rate=lr_rate)
model1 = Sequential() 
model1.add(Dense(32, activation='relu', input_shape=(x_train.shape[1],), name='base1')) #,kernel_regularizer='l2'
model1.add(Dense(16, activation='relu', name='base2'))
model1.add(Dense(10, activation='relu', name='base3'))    
model1.add(Dense(1, activation='sigmoid', name='personal'))
#　model1.compile(optimizer=opt_adam, loss=tf.losses.BinaryFocalCrossentropy(gamma=2.0), metrics=METRICS)

model2 = Sequential() 
model2.add(Dense(32, activation='relu', input_shape=(x_train.shape[1],), name='base1')) #,kernel_regularizer='l2'
model2.add(Dense(16, activation='relu', name='base2'))
model2.add(Dense(10, activation='relu', name='base3'))    
model2.add(Dense(1, activation='sigmoid', name='personal'))
# model2.compile(optimizer=opt_adam, loss=tf.losses.BinaryFocalCrossentropy(gamma=2.0), metrics=METRICS)
parameter = model2.get_weights()[:6]
for layer_name in ['base1', 'base2', 'base3']:
    start_index = (int(layer_name[-1])-1)*2
    end_index = int(layer_name[-1])*2
    model1.get_layer(layer_name).set_weights(parameter[start_index:end_index])

'''print(model1.get_layer('personal').get_weights())
model1.get_layer('personal').set_weights(model2.get_weights()[6:])
print(model2.get_layer('personal').get_weights())
print(model1.get_layer('personal').get_weights())'''

# print(model1.get_layer('personal').get_weights())
personal_layer = Dense(1, activation='sigmoid', name='personal')
personal_layer.build((1, 10))
# print(personal_layer.get_weights())
model1.get_layer('personal').set_weights(personal_layer.get_weights())
#　print(model1.get_layer('personal').get_weights())


hist_dict = {'loss': [0.07672514021396637, 0.04929312318563461, 0.04727593809366226, 0.04546748474240303, 0.044543664902448654], 'precision_1': [0.040928132832050323, 0.040928132832050323, 0.040928132832050323, 0.04312115162611008, 0.050862450152635574], 'recall_1': [1.0, 1.0, 1.0, 0.9921259880065918, 0.9055117964744568], 'auc_1': [0.4102994501590729, 0.4500611424446106, 0.6463955640792847, 0.7293267250061035, 0.7272550463676453], 'val_loss': [0.050409842282533646, 0.048841677606105804, 0.04650086537003517, 0.044900018721818924, 0.043144889175891876], 'val_precision_1': [0.04123711213469505, 0.04123711213469505, 0.04123711213469505, 0.0920245423913002, 0.055956680327653885], 'val_recall_1': [1.0, 1.0, 1.0, 0.9375, 0.96875], 'val_auc_1': [0.2578335106372833, 0.7487819194793701, 0.7763356566429138, 0.7806829214096069, 0.7778058052062988]} 
print(hist_dict)
print(list(hist_dict.keys())[0])
# print(hist_dict[hist_dict.keys()[0]])