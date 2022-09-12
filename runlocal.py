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
from tensorflow.keras.optimizers import Adam, Adagrad
from sklearn.metrics import roc_auc_score
import utils

parser = argparse.ArgumentParser(description="Flower")
parser.add_argument("--hospital", type=int, choices=range(0, 10), required=True)
parser.add_argument("--seed", type=int, choices=range(0, 1000), required=True)
parser.add_argument("--seer", type=int, required=True)
parser.add_argument("--encode_dict", type=str, required=True)   # put weight or average
args = parser.parse_args()

#SEED

lr_rate = 0.001
epoch = 100
size=0.2
auc_val_result = {}
hospital_list = [2,3,6,8,9]

set_thres=0.19
METRICS = [
      metrics.Precision(thresholds=set_thres),
      metrics.Recall(thresholds=set_thres),
      metrics.AUC()
]

site_id = args.hospital
seed = args.seed
seer = args.seer
average_weight = args.encode_dict

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

imputation = utils.imputation()
enc_dict = utils.choose_dict(average_weight,seer)
mapping = utils.mapping()
split = utils.split_data(size,seed)

if seer == 1:
    columns = ["Class","LOC", "Gender", "Age", "AJCCstage", "DIFF", "LYMND", "TMRSZ", "SSF1", "SSF2"]
    
    if site_id != 9:
        df = pd.read_csv(r'/home/refu0917/lungcancer/server/AllCaseCtrl_final.csv')
        df = df[columns]
        df['Class'] = df['Class'].apply(lambda x:1 if x != 0 else 0)
        df = df[df['LOC'] == site_id]
        dfimp = df.fillna(9)
        dfmap = mapping(enc_dict(),dfimp)

    elif site_id == 9:
        df = pd.read_csv(r'/home/refu0917/lungcancer/data/seerdb.csv',index_col = [0])
        df = df[columns]
        df = df[df['LOC'] == site_id]
        dfmap = mapping(enc_dict(),df)

elif seer == 0:
    columns = ["Class","LOC", "FullDate","Gender", "Age", "CIG",
            "ALC", "BN", "MAGN", "AJCCstage", "DIFF", "LYMND",
            "TMRSZ", "OP", "RTDATE", "STDATE", "BMI_label",
            "SSF1", "SSF2", "SSF3", "SSF4", "SSF6"]
    df = pd.read_csv(r'/home/refu0917/lungcancer/server/AllCaseCtrl_final.csv')
    df = df[columns]
    df["Class"] = df['Class'].apply(lambda x:1 if x != 0 else 0)
    df = df[df['LOC'] == site_id]
    dfimp = imputation(df, 'drop_and_fill')
    dfmap = mapping(enc_dict(),dfimp)

x_train,x_test,y_train,y_test = split(dfmap)

opt_adam = Adam(learning_rate=lr_rate)
model = Sequential() 
model.add(Dense(32, activation='relu', input_shape=(x_train.shape[1],))) #,kernel_regularizer='l2'
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='relu'))    
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=opt_adam, loss=tf.losses.BinaryFocalCrossentropy(gamma=2.0), metrics=METRICS)


hist = model.fit(x_train,y_train,batch_size=16,epochs=epoch,verbose=2,validation_data=(x_test, y_test))
y_pred = model.predict(x_test)
#auc_val_result[str(site_id)] = [roc_auc_score(y_test, y_pred)]
val_df = pd.read_csv('./output_folder/local_val_df_average.csv', index_col=[0])
val_df.loc[seed,f'site{site_id}'] = roc_auc_score(y_test, y_pred)
val_df.to_csv('./output_folder/local_val_df_average.csv')
print(f'AUC by sklearn : {roc_auc_score(y_test,y_pred)}')

# Use Local model to evaluate other hospital : 
'''hospital_list.remove(args.hospital)
for i in hospital_list:
  _,x_val,_,y_val = load_data(i)
  y_pred_val = model.predict(x_val)
  auc_val_result[str(i)] = [roc_auc_score(y_val, y_pred_val)]
auc_val_result = dict(sorted(auc_val_result.items()))  # Sort the AUC result dict
print(auc_val_result)'''

#with open(f'./data_folder/Local_AUC_val_{args.hospital}.pickle', 'wb') as f:
#    pickle.dump(auc_val_result, f)

#with open(f'./data_folder/Local_AUC_{args.hospital}.pickle', 'wb') as f:
#    pickle.dump(hist.history['val_auc'], f)

#local_auc_df = pd.read_csv(f'./data_folder/local_auc_df{args.hospital}.csv')
#local_auc_df[f'seed{seed}'] = hist.history['val_auc']
#local_auc_df.to_csv(f'./data_folder/local_auc_df{args.hospital}.csv',index=False)

'''local_auc_df = pd.DataFrame(data = hist.history['val_auc'], columns=[f'seed{seed}'])
local_auc_df.to_csv(f'local_auc_df{args.hospital}.csv',index=False)'''
