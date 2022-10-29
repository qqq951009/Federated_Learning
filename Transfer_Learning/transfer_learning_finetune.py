import os
import keras
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
parser.add_argument("--hospital", type=int, choices=range(0, 1000), required=True)
parser.add_argument("--seer", type=int, required=True)
args = parser.parse_args()

size = 0.2
seed = args.seed
seer = args.seer
hospital = args.hospital
dir_name = '/home/refu0917/lungcancer/remote_output1/output_folder/imputation_test_folder/'
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

if seer == 1:
  columns = ["Class","LOC", "FullDate", "Gender", "Age", "AJCCstage", 
             "DIFF", "LYMND", "TMRSZ", "SSF1", "SSF2", "SSF4", "OP"]
  output_file_name = 'transfer_learning_score_seer.csv'

elif seer == 0:
  columns = ["Class","LOC", "FullDate","Gender", "Age", "CIG",
            "ALC", "BN", "MAGN", "AJCCstage", "DIFF", "LYMND",
            "TMRSZ", "OP", "RTDATE", "STDATE", "BMI_label",
            "SSF1", "SSF2", "SSF3", "SSF4", "SSF6"]
  output_file_name = 'transfer_learning_score.csv'

with open('imputationdf.pickle', 'rb') as f:
    site_imp_dict = pickle.load(f)
with open('mapping.pickle', 'rb') as f:
    site_map_dict = pickle.load(f)

def main() -> None:

    map = utils.mapping()

    # Select the hospital from the dataframe after imputation
    dfimp = site_imp_dict[hospital]
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
    model = keras.models.load_model('pretrained_model')
    model.fit(x_train,y_train,batch_size=16,epochs=100,verbose=2,validation_data=(x_test, y_test))
    y_pred = model.predict(x_test)
    score_df = pd.read_csv(dir_name + output_file_name,index_col=[0])
    score_df.loc[seed,f"site{hospital}"] = roc_auc_score(y_test, y_pred)
    score_df.to_csv(dir_name + output_file_name)

if __name__ == "__main__":
    main()