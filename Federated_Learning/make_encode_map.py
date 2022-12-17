import os
import yaml
import time
import pickle
import random
import argparse
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import utils

with open('../config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.Loader)

parser = argparse.ArgumentParser(description="Flower")
parser.add_argument("--seed", type=int, choices=range(0, 1000), required=True)
parser.add_argument("--seer", type=int, default=0)
args = parser.parse_args()
seed = args.seed
seer = args.seer
size = 0.2
index_list, df_list = [], []

if seer == 1:
  columns = ["Class","LOC", "FullDate", "Gender", "Age", "AJCCstage", 
             "DIFF", "LYMND", "TMRSZ", "SSF1", "SSF2", "SSF4", "OP"]
  df1 = pd.read_csv(r'/home/refu0917/lungcancer/server/AllCaseCtrl_final.csv')
  df2 = pd.read_csv(r'/home/refu0917/lungcancer/data/seerdb.csv',index_col = [0])
  df1, df2 = df1[columns], df2[columns]
  df = pd.concat([df1, df2], axis = 0)
  df['Class'] = df['Class'].apply(lambda x:1 if x != 0 else 0)


elif seer == 0:
  columns = ["Class","LOC", "Gender", "Age", "CIG",
            "ALC", "BN", "MAGN", "AJCCstage", "DIFF", "LYMND",
            "TMRSZ", "OP", "RTDATE", "STDATE", "BMI_label",
            "SSF1", "SSF2", "SSF3", "SSF4", "SSF6"] #"FullDate",
  df = pd.read_csv(config['data_dir']['8hos'],index_col=[0])
  df = df[columns]
  # df['Class'] = df['Class'].apply(lambda x:1 if x != 0 else 0)

for i in np.unique(df.LOC):
    tempdf = df[df['LOC'] == i]
    df_list += [tempdf]
    index_list.append(i)

make_map_fn = utils.make_map(seed, size)
make_map_fn(df_list, index_list, df, columns, config['imp_method'])