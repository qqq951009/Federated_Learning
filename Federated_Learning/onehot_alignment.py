import os
import yaml
import pickle
import random
import argparse
import pandas as pd
import utils
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description="Flower")
parser.add_argument("--seed", type=int, default=42, choices=range(0, 1000))
parser.add_argument("--hospital", type=int, choices=range(0, 1000), required=True)
args = parser.parse_args()
seed = args.seed
site = args.hospital

with open('../config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.Loader)
size = config['test_size']

columns = ["Class","LOC", "Gender", "Age", "CIG",
            "ALC", "BN", "MAGN", "AJCCstage", "DIFF", "LYMND",
            "TMRSZ", "OP", "RTDATE", "STDATE", "BMI_label",
            "SSF1", "SSF2", "SSF3", "SSF4", "SSF6"]
df = pd.read_csv(config['data_dir']['8hos'],index_col=[0])
df = df[columns]

x_train, y_train, x_test, y_test = utils.onehot_aligment(df, seed, site, config)
print(x_train)