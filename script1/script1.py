

import pandas as pd
import numpy as np

#Create a Dictionary of series
dico = {'Name':pd.Series(['MerouBrun','Plie']),
   'Age':pd.Series([4,3]),
   'Taille':pd.Series([40,25])}

#Create a DataFrame
df = pd.DataFrame(dico)
print(df)

print("------")
for e in df.itertuples():
    print(df['Name'])

    
print("------")
print(df.iat[1,0])