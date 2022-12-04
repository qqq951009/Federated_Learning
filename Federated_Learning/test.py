import pandas as pd
import pickle
import utils

with open('./encode_dict_folder/imputationdf.pickle', 'rb') as f:
    site_imp_dict = pickle.load(f)
with open('./encode_dict_folder/mapping.pickle', 'rb') as f:
    site_map_dict = pickle.load(f)
    
columns = ["Class","LOC", "FullDate","Gender", "Age", "CIG",
            "ALC", "BN", "MAGN", "AJCCstage", "DIFF", "LYMND",
            "TMRSZ", "OP", "RTDATE", "STDATE", "BMI_label",
            "SSF1", "SSF2", "SSF3", "SSF4", "SSF6"]

    
   
    
map = utils.mapping()
split_train_test = utils.split_data()

# Select the hospital from the dataframe after imputation
dfimp = site_imp_dict[2]
trainimp, testimp = dfimp['train'],dfimp['test']

# Map the target encoding
trainenc = map(site_map_dict, trainimp, columns[3:])
testenc = map(site_map_dict, testimp, columns[3:])
print(trainenc.head())
print(trainenc.DIFF.value_counts())
print(trainenc.SSF3.value_counts())
# print(trainenc.Class.value_counts())
#trainenc['Class'] = trainenc['Class'].apply(lambda x:2 if x!=1 else 1)
#testenc['Class'] = testenc['Class'].apply(lambda x:2 if x!=1 else 1)