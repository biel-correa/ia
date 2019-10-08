import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

df =pd.read_csv('/home/gabriel/Documentos/github/udemy/deveres/dever1/arquivos/titanic-train.csv')
scatter_matrix(df,alpha=0.3,figsize=(6,6))

plt.show()