import pandas as pd
import numpy as np
from category_encoders import TargetEncoder
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/horse-colic.csv'
df = pd.read_csv(url, header=None, na_values='?')
print(df.head())

data = df.values
ix = [i for i in range(data.shape[1]) if i != 23]
print(ix)
x, y = data[:, ix], data[:, 23]
print(x)
print(y)