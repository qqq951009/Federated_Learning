import collections
import pickle
import pandas as pd
from utils import drop_year
import utils
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

columns = ["Class","LOC", "FullDate","Gender", "Age", "CIG",
        "ALC", "BN", "MAGN", "AJCCstage", "DIFF", "LYMND",
        "TMRSZ", "OP", "RTDATE", "STDATE", "BMI_label",
        "SSF1", "SSF2", "SSF3", "SSF4", "SSF6"]
df = pd.read_csv(r'/home/refu0917/lungcancer/server/AllCaseCtrl_final.csv')
df = df[columns]
df['Class'] = df['Class'].apply(lambda x:1 if x != 0 else 0)

df2 = df[df['LOC'] == 2]

iterative_imputation_fn = utils.iterative_imputation()
drop_year_fn = utils.drop_year()
df2 = drop_year_fn(df2)
trainset, testset = train_test_split(df2,test_size = 0.2,stratify=df2['Class'],random_state=42)
imputer = IterativeImputer(random_state=42, estimator=RandomForestClassifier(),initial_strategy = 'most_frequent')
df_imp = imputer.fit_transform(trainset)
df_imp = df_imp.astype(int)
df_imp = pd.DataFrame(data = df_imp,columns = df2.columns)


#trainimp = iterative_imputation_fn(trainset, 42)
print(df_imp.SSF6.value_counts())
print(df_imp.SSF4.value_counts())