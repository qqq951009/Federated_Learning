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
import yaml
import utils
import os 

with open('../config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.Loader)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
parser = argparse.ArgumentParser(description="Flower")
parser.add_argument("--seed", type=int, choices=range(0, 1000), required=True)
parser.add_argument("--seer", type=int, default=0)
args = parser.parse_args()

#SEED
seed = args.seed
seer = args.seer
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

size = 0.2
eval_siteID = 2
lr_rate = 0.001
set_thres = 0.19
rounds = 20

METRICS = [
      metrics.Precision(thresholds=set_thres),
      metrics.Recall(thresholds=set_thres),
      metrics.AUC()
]


if seer == 1:
    # min_client = 5
    df1 = pd.read_csv(r'/home/refu0917/lungcancer/server/AllCaseCtrl_final.csv')
    df2 = pd.read_csv(r'/home/refu0917/lungcancer/data/seerdb.csv',index_col = [0])
    df = pd.concat([df1, df2], axis = 0)
    # min_client = len(np.unique(df.LOC))
    columns = ["Class","LOC", "FullDate", "Gender", "Age", "AJCCstage", 
                "DIFF", "LYMND", "TMRSZ", "SSF1", "SSF2", "SSF4", "OP"]
elif seer == 0:
    df = pd.read_csv(config['data_dir']['8hos'],index_col=[0])
    min_client = len(np.unique(df.LOC))
    # min_client = 2
    columns = ["Class","LOC", "Gender", "Age", "CIG",
                "ALC", "BN", "MAGN", "AJCCstage", "DIFF", "LYMND",
                "TMRSZ", "OP", "RTDATE", "STDATE", "BMI_label",
                "SSF1", "SSF2", "SSF3", "SSF4", "SSF6"]# "FullDate",
  

with open('./encode_dict_folder/imputationdf.pickle', 'rb') as f:
    site_imp_dict = pickle.load(f)
with open('./encode_dict_folder/mapping.pickle', 'rb') as f:
    site_map_dict = pickle.load(f)


def main() -> None:

    map = utils.mapping()
    split_train_test = utils.split_data()

    # Select the biggest site from the dataframe after imputation
    dfimp = site_imp_dict[2]
    trainimp, testimp = dfimp['train'],dfimp['test']

    # Map the target encoding
    trainenc = map(site_map_dict, trainimp, columns[2:])
    testenc = map(site_map_dict, testimp, columns[2:])
    trainenc['Class'] = trainenc['Class'].apply(lambda x:1 if x!=1 else 0)
    testenc['Class'] = testenc['Class'].apply(lambda x:1 if x!=1 else 0)

    # Split X and Y
    x_train,y_train = trainenc.drop(columns = ['Class', 'LOC']), trainenc['Class']
    x_test, y_test = testenc.drop(columns = ['Class', 'LOC']), testenc['Class']

    opt_adam = Adam(learning_rate=lr_rate)
    model = Sequential() 
    model.add(Dense(32, activation='relu', input_shape=(x_train.shape[1],), name='base1')) #,kernel_regularizer='l2'
    model.add(Dense(16, activation='relu', name='base2'))
    model.add(Dense(10, activation='relu', name='base3'))    
    model.add(Dense(1, activation='sigmoid', name='personal'))
    # model.compile(optimizer=opt_adam, loss=tf.losses.BinaryFocalCrossentropy(gamma=2.0), metrics=METRICS)

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
    fl.server.start_server("[::]:6000", config={"num_rounds":rounds}, strategy=strategy)


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
