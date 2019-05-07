import numpy as np
import pandas as pd

name = np.array(['ana','beto','caio','denis'])
age = np.array([10,15,22,12])

df = pd.DataFrame(data=dict(name=name, age=age))
df.to_csv('test_df.csv', sep='\t')
