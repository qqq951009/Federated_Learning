from typing import Any, Callable, Dict, List, Optional, Tuple
import random
import pickle
import argparse
import flwr as fl
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import metrics
import tensorflow_addons as tfa
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from typing import Any, Callable, Dict, List, Optional, Tuple
from category_encoders import TargetEncoder, LeaveOneOutEncoder
from tensorflow.keras.layers import Dense, Embedding, Flatten,Dropout
import configparser
import utils

config = configparser.ConfigParser()
config.read('config.ini')


parser = argparse.ArgumentParser(description="Flower")
parser.add_argument("--seed", type=int, choices=range(0, 1000), required=True)
parser.add_argument("--seer", type=int, required=True)
args = parser.parse_args()

#SEED
seed = args.seed
seer = args.seer
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
lr_rate = 0.001
set_thres = 0.19
rounds = 20
server_evaluate = {
    "loss":[],
    "AUC":[],
    "Recall":[],
    "Precision":[]
}
METRICS = [
      metrics.Precision(thresholds=set_thres),
      metrics.Recall(thresholds=set_thres),
      metrics.AUC()
]

imputation = utils.imputation()
enc_dict = utils.choose_dict('average',seer)
mapping = utils.mapping()
split = utils.split_data(0.2,seed)


df = pd.read_csv(r'/home/refu0917/lungcancer/server/AllCaseCtrl_final.csv')

if seer == 1:
  columns = ["Class","LOC", "Gender", "Age", "AJCCstage", "DIFF", "LYMND", "TMRSZ", "SSF1", "SSF2"]
  df = df[columns]
  df['Class'] = df['Class'].apply(lambda x:1 if x != 0 else 0)
  df = df[df['LOC'] == 2]
  dfimp = df.fillna(9)
  dfmap = mapping(enc_dict(),dfimp)
  min_client = 5

elif seer == 0:
  columns = ["Class","LOC", "FullDate","Gender", "Age", "CIG",
            "ALC", "BN", "MAGN", "AJCCstage", "DIFF", "LYMND",
            "TMRSZ", "OP", "RTDATE", "STDATE", "BMI_label",
            "SSF1", "SSF2", "SSF3", "SSF4", "SSF6"]
  df = df[columns]
  df["Class"] = df['Class'].apply(lambda x:1 if x != 0 else 0)
  df = df[df['LOC'] == 2]
  dfimp = imputation(df, 'drop_and_fill')
  dfmap = mapping(enc_dict(),dfimp)
  min_client = 4

x_train,x_test,y_train,y_test = split(dfmap)


def main() -> None:
    opt_adam = Adam(learning_rate=lr_rate)
    model = Sequential() 
    model.add(Dense(32, activation='relu', input_shape=(x_train.shape[1],))) #,kernel_regularizer='l2'
    model.add(Dense(16, activation='relu'))
    model.add(Dense(10, activation='relu'))    
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=opt_adam, loss=tf.losses.BinaryFocalCrossentropy(gamma=2.0), metrics=METRICS)

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=min_client,
        min_eval_clients=min_client,
        min_available_clients=min_client,
        #eval_fn=get_eval_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.weights_to_parameters(model.get_weights()),
    )

    # Start Flower server for four rounds of federated learning
    fl.server.start_server("[::]:7000", config={"num_rounds":rounds}, strategy=strategy)


'''def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` 
    # The `evaluate` function will be called after every round
    def evaluate(
        weights: fl.common.Weights,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:

        # Update model with the latest parameters
        model.set_weights(weights)  
        loss,precision,recall,auc = model.evaluate(x_test, y_test)
        
        y_pred = model.predict(x_test)
        
        return loss, {"AUC": roc_auc_score(y_test, y_pred)}

    return evaluate'''


def fit_config(rnd: int):
    config = {
        "rnd": rnd,
        "batch_size": 16,
        "local_epochs": 5
    }
    return config

def evaluate_config(rnd: int):
    val_steps = 5 if rnd < 4 else 10
    return {"val_steps": val_steps, "rnd" : rnd}


if __name__ == "__main__":
    main()
