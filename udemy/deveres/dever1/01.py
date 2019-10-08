import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('/home/gabriel/Documentos/github/udemy/deveres/dever1/arquivos/international-airline-passengers.csv')
print(df.info() , '\n')
print(df.head() , '\n')
df['Month'] = pd.to_datetime(df['Month'])
df = df.set_index('Month')
plt.plot(df)
plt.show()