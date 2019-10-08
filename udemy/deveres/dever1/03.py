import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('/home/gabriel/Documentos/github/udemy/deveres/dever1/arquivos/weight-height.csv')
males = df[df['Gender'] == 'Male']
females = df.query('Gender == "Female"')
# fig,ax = plt.subplot()

males['Height'].plot(
    kind='hist',
    bins=50,
    title='Histogram',
    alpha = 0.6,
    figsize=(5,5),
    color='blue'
)

females['Height'].plot(
    kind='hist',
    bins=50,
    title='Histogram',
    alpha = 0.6,
    figsize=(5,5),
    color='red'
)

plt.title('Distribuicao de altura')
plt.legend(['Homens','Mulheres'])
plt.xlabel('Altura')

plt.axvline(males['Height'].mean(),color='blue',linewidth=2)
plt.axvline(females['Height'].mean(),color='red',linewidth=2)

plt.show()
