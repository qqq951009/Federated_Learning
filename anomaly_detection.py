import pickle
import random
import argparse
import keras
import flwr as fl
import numpy as np
import pandas as pd
import configparser
import tensorflow as tf
from keras import metrics
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten,Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_auc_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.utils import class_weight
import utils


columns = ["Class","LOC", "FullDate","Gender", "Age", "CIG",
            "ALC", "BN", "MAGN", "AJCCstage", "DIFF", "LYMND",
            "TMRSZ", "OP", "RTDATE", "STDATE", "BMI_label",
            "SSF1", "SSF2", "SSF3", "SSF4", "SSF6"]
set_thres = 0.5
METRICS = [
        metrics.TruePositives(name='tp'),
        metrics.FalsePositives(name='fp'),
        metrics.TrueNegatives(name='tn'),
        metrics.FalseNegatives(name='fn'),
        metrics.Precision(thresholds=set_thres), # thresholds=set_thres
        metrics.Recall(thresholds=set_thres), # thresholds=set_thres
        metrics.AUC(),
        metrics.AUC(name='prc', curve='PR')]

imputation = utils.imputation()
iterative_imputation = utils.iterative_imputation()
enc_dict = utils.choose_dict('average',0)
mapping = utils.mapping()
split = utils.split_data(0.2,43)

df = pd.read_csv(r'/home/refu0917/lungcancer/server/AllCaseCtrl_final.csv')
df = df[columns]
df["Class"] = df['Class'].apply(lambda x:1 if x != 0 else 0)
df = df[df['LOC'] == 3]
df_imp = iterative_imputation(df)
df_imp = pd.DataFrame(data=df_imp, columns=columns[:1]+columns[3:])
df_map = mapping(enc_dict(),df_imp)
x_train,x_test,y_train,y_test = split(df_map)

sklearn_weight = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y = y_train)
weight_dict = {0:sklearn_weight[0], 1: sklearn_weight[1]}
svc = SVC(kernel='linear', class_weight=weight_dict)
lr = LogisticRegression(penalty='l2', class_weight=weight_dict)

svc.fit(x_train,y_train)
lr.fit(x_train,y_train)

y_pred_svc = svc.predict(x_test)
y_pred_lr = lr.predict(x_test)

print('scv')
print(roc_auc_score(y_test,y_pred_svc), precision_score(y_test, y_pred_svc))
print('lr')
print(roc_auc_score(y_test,y_pred_lr), precision_score(y_test, y_pred_lr))

train_normal_index = y_train[y_train == 0].index
train_anomaly_index = y_train[y_train == 1].index 
test_normal_index = y_test[y_test == 0].index
test_anomaly_index = y_test[y_test == 1].index 

x_train_normal = x_train.loc[train_normal_index]
x_train_anomaly = x_train.loc[train_anomaly_index]
x_test_normal = x_test.loc[test_normal_index]
x_test_anomaly = x_test.loc[test_anomaly_index]

input_size = len(x_train)
intermidiate_size1 = 16
intermidiate_size2 = 8
code_size = 2

class AnomalyDetector(tf.keras.Model):
  def __init__(self):
    super(AnomalyDetector, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(16, activation="relu"),
      layers.Dense(8, activation="relu"),
      layers.Dense(2, activation="relu")])

    self.decoder = tf.keras.Sequential([
      layers.Dense(8, activation="relu"),
      layers.Dense(16, activation="relu"),
      layers.Dense(19, activation="sigmoid")])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
autoencoder = AnomalyDetector()


autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy')
autoencoder.fit(x_train_normal, x_train_normal, epochs=100, batch_size=32, shuffle=True, validation_data=(x_test, x_test))

reconstructions_normal = autoencoder.predict(x_train_normal)
normal_loss = tf.keras.losses.mae(reconstructions_normal, x_train_normal)

reconstructions_anomaly = autoencoder.predict(x_train_anomaly)
anomaly_loss = tf.keras.losses.mae(reconstructions_anomaly, x_train_anomaly)


plt.hist(normal_loss[None, :], bins=20)
plt.hist(anomaly_loss[None, :], bins=20)
plt.xlabel("Train loss")
plt.ylabel("No of examples")
plt.savefig('plot.png')