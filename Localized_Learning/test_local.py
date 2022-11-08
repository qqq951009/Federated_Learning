import pandas as pd
import numpy as np
from category_encoders import TargetEncoder

data = {'Temperature': ['Hot','Cold','Very Hot','Warm','Hot','Warm','Warm','Hot','Hot','Cold'],
        'Color': ['Red','Yellow','Blue','Blue','Red','Yellow','Red','Yellow','Yellow','Yellow'],
        'Class':[1,1,1,0,1,0,1,0,1,1]}
df = pd.DataFrame(data, columns = ['Temperature', 'Color', 'Class'])


x = df.drop(['Class'], axis=1)
y = df['Class']

encoder = TargetEncoder(cols=['Temperature', 'Color'])
encoder.fit(x,y)
encoder.transform(x, y)