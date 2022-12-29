import pickle

with open('/home/refu0917/lungcancer/remote_output1/encode_dict_folder/imputationdf.pickle', 'rb') as f:
    temp = pickle.load(f)

print(temp[2])