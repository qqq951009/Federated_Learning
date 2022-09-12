import pandas as pd
import os


df = pd.DataFrame(columns = ['site2','site3','site6','site8'])
df2 = pd.read_csv('/home/refu0917/lungcancer/remote_output1/output_folder/fl_folder/df_fedavg_average2.csv', index_col=[0])
df3 = pd.read_csv('/home/refu0917/lungcancer/remote_output1/output_folder/fl_folder/df_fedavg_average3.csv', index_col=[0])
df6 = pd.read_csv('/home/refu0917/lungcancer/remote_output1/output_folder/fl_folder/df_fedavg_average6.csv', index_col=[0])
df8 = pd.read_csv('/home/refu0917/lungcancer/remote_output1/output_folder/fl_folder/df_fedavg_average8.csv', index_col=[0])

df = pd.concat([df2, df3, df6, df8],axis=1)
    
df.to_csv('/home/refu0917/lungcancer/remote_output1/output_folder/df_fedavg_average.csv')