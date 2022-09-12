import os
import time
import pickle
import random
import argparse
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import utils

parser = argparse.ArgumentParser(description="Flower")
parser.add_argument("--seer", type=int, required=True)
args = parser.parse_args()

weight_dict, index_list, df_list = {}, [], []

if args.seer == 1:
    columns = ["Class","LOC", "Gender", "Age", "AJCCstage", "DIFF", "LYMND", "TMRSZ", "SSF1", "SSF2"]
    dfseer = pd.read_csv(r'/home/refu0917/lungcancer/data/seerdb.csv',index_col = [0])
    df9 = dfseer[columns]
    index_list = [9,2,3,6,8]
    df_list.append(df9)
    
elif args.seer == 0:
    columns = ["Class","LOC", "FullDate","Gender", "Age", "CIG",
            "ALC", "BN", "MAGN", "AJCCstage", "DIFF", "LYMND",
            "TMRSZ", "OP", "RTDATE", "STDATE", "BMI_label",
            "SSF1", "SSF2", "SSF3", "SSF4", "SSF6"]
    index_list = [2,3,6,8]


df = pd.read_csv('/home/refu0917/lungcancer/server/AllCaseCtrl_final.csv')
df = df[columns]
df['Class'] = df['Class'].apply(lambda x:1 if x != 0 else 0)
df2 = df[df['LOC'] == 2]
df3 = df[df['LOC'] == 3]
df6 = df[df['LOC'] == 6]
df8 = df[df['LOC'] == 8]
df_list += [df2,df3,df6,df8]


for i,j in zip(index_list,df_list):
    weight_dict[i] = len(j)/sum([len(i) for i in df_list]) # total length

encode_dict1 = utils.encode_dict(weight_dict, df_list, index_list, columns)
encode_dict2 = utils.encode_dict(weight_dict, df_list, index_list, columns)
encode_map_average = encode_dict1('average')
encode_map_weight = encode_dict2('weight_average')
print(encode_map_average)
print(encode_map_weight)

with open('./encode_dict_folder/encode_average_seer.pickle', 'wb') as output:
    pickle.dump(encode_map_average, output)

with open('./encode_dict_folder/encode_weight_seer.pickle', 'wb') as output:
    pickle.dump(encode_map_weight, output)
