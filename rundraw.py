import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Draw Validation Heatmap

local_val_data = pd.DataFrame()
for i in [2,3,6,8]:
    with open(f'./data/Local_AUC_val_{i}.pickle', 'rb') as f:
        temp = pickle.load(f)
        index = 'model'+str(i)
        temp = pd.DataFrame(temp,index = [index])
    local_val_data = pd.concat([local_val_data,temp])
local_val_data.rename(columns={'2': 'val2','3': 'val3','6': 'val6','8': 'val8'}, inplace=True)


'''fl_val_data = pd.DataFrame()
for i in [2,3,6,8]:
    with open(f'FL_AUC_val_{i}.pickle', 'rb') as f:
        temp = pickle.load(f)
        index = 'model'+str(i)
        temp = pd.DataFrame(temp,index = [index])
    fl_val_data = pd.concat([fl_val_data,temp])
fl_val_data.rename(columns={'2': 'val2','3': 'val3','6': 'val6','8': 'val8'}, inplace=True)'''

fl_val_data = pd.DataFrame()
with open(f'./data/FL_AUC_val_2.pickle', 'rb') as f:
    temp = pickle.load(f)
    index = 'model'+str(i)
    fl_val_data = pd.DataFrame(temp,index = ['FLmodel'])

plt.figure(figsize=(20,16))
plt.subplot(3,2,1)
ax1 = sns.heatmap(fl_val_data, xticklabels=True,annot=True, yticklabels=True, linewidths=.5,fmt='.4g',annot_kws={"size": 18},)  
ax1.set_title('FL Validation Heatmap')
plt.yticks(rotation=0)
plt.subplot(3,2,2)
ax2 = sns.heatmap(local_val_data, xticklabels=True,annot=True, yticklabels=True, linewidths=.5,fmt='.4g',annot_kws={"size": 18},)  
ax2.set_title('Local Validation Heatmap')
plt.yticks(rotation=0)

for i,j in enumerate([2,3,6,8]):
    scoredf = pd.DataFrame()
    with open(f'./data/Local_AUC_{j}.pickle', 'rb') as f:
        local_score = pickle.load(f)
    with open(f'./data/FL_AUC_{j}.pickle', 'rb') as f:
        fl_score = pickle.load(f)
    scoredf['Local'] = local_score
    scoredf['FL'] = fl_score
    sns.set_style("darkgrid")
    plt.subplot(3,2,i+3) 
    ax = sns.lineplot(data=scoredf,palette=['g','r'])
    ax.set_title(f'hospital{j}')
plt.savefig(f'heatmap.png')


