import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('/home/gabriel/Documentos/github/udemy/deveres/dever1/arquivos/weight-height.csv')
print(df.info(), '\n')
print(df.head(), '\n')
print(df.describe(), '\n')

df['Gender'].value_counts()

# _=df.plot(kind='scatter',x='Height',y='Weight')
males=df[df['Gender'] =='Male']
females = df.query('Gender == "Female"')
fig, ax =plt.subplots()

males.plot(
    kind='scatter',
    x='Height',
    y='Weight',
    ax=ax,
    color='blue',
    alpha=0.2,
    title=('Populacao de Homens e Mulheres')
)

females.plot(
    kind='scatter',
    x='Height',
    y='Weight',
    ax=ax,
    color='red',
    alpha=0.2
)

plt.show()