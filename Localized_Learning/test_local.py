import utils
import pandas as pd
from sklearn.model_selection import train_test_split

seer = 0
size=0.2
seed=42
site_id=2
map = utils.mapping()
drop_year = utils.drop_year()
iterative_imputation = utils.iterative_imputation()
target_encode = utils.target_encoding(False)
train_enc_map_fn = utils.train_enc_map()

if seer == 1:
    columns = ["Class","LOC", "FullDate", "Gender", "Age", "AJCCstage", 
                "DIFF", "LYMND", "TMRSZ", "SSF1", "SSF2", "SSF4", "OP"]
    df1 = pd.read_csv(r'/home/refu0917/lungcancer/server/AllCaseCtrl_final.csv')
    df2 = pd.read_csv(r'/home/refu0917/lungcancer/data/seerdb.csv',index_col = [0])
    df1["Class"], df2["Class"] = df1['Class'].apply(lambda x:2 if x != 0 else 1), df2['Class'].apply(lambda x:2 if x != 0 else 1)
    df1, df2 = df1[columns], df2[columns]
    df = pd.concat([df1, df2])

elif seer == 0:
    columns = ["Class","LOC", "FullDate","Gender", "Age", "CIG",
            "ALC", "BN", "MAGN", "AJCCstage", "DIFF", "LYMND",
            "TMRSZ", "OP", "RTDATE", "STDATE", "BMI_label",
            "SSF1", "SSF2", "SSF3", "SSF4", "SSF6"]
    df = pd.read_csv(r'/home/refu0917/lungcancer/server/AllCaseCtrl_final.csv')
    df = df[columns]

df['Class'] = df['Class'].apply(lambda x:2 if x != 0 else 1)
df = df[df['LOC'] == site_id]

# Drop the year smaller than 2010
df = drop_year(df)
# Split df into train and test set
trainset, testset = train_test_split(df,test_size = size,stratify=df['Class'],random_state=seed)
# Impute the trainset and testset respectively
trainimp = iterative_imputation(trainset, seed)
testimp = iterative_imputation(testset, seed)
print(trainimp[trainimp['Class'] == 1]['Gender'].value_counts())
print(trainimp[trainimp['Class'] == 2]['Gender'].value_counts())
# Encode trainset and map the encode dictionary to testset
trainenc = target_encode(trainimp)
train_enc_dict = train_enc_map_fn(trainenc,trainimp, columns[3:],df)
print(train_enc_dict)
testenc = map(train_enc_dict, testimp, columns[3:])