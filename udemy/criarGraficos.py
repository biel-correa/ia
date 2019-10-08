import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data1 = np.random.normal(0,0.1,1000)
data2 = np.random.normal(1,0.4,1000) + np.linspace(0,1,1000)
data3 = 2 + np.random.random(1000) * np.linspace(1,5,1000)
data4 = np.random.normal(3,0.2,1000) + 0.3 * np.sin(np.linspace(0,20,1000))

data = np.vstack([data1,data2,data3,data4]).transpose()

df = pd.DataFrame(data,columns=['data1','data2','data3','data4',])
# print(df.head())

# df.plot(title='Line Plot')

# plt.plot(df)
# plt.title('Line Plot')
# plt.legend(['data1','data2','data3','data4',])

# df.plot(style='.')

# _=df.plot(kind='scatter', x='data1',y='data2',xlim=(-1.5,1.5),ylim=(0,3))

df.plot(
    kind='hist',
    bins=50,
    title='Histogram',
    alpha = 0.6,
    figsize=(5,5)
)

# df.plot(
#     kind='hist',
#     bins=100,
#     title='Cumulative distributions',
#     normed=True,
#     cumulative=True,
#     alpha=0.4,
#     figsize=(6,4)
# )

# df.plot(
#     kind='box',
#     title='Boxplot'
# )

# fig,ax=plt.subplots(2,2,figsize=(12,9))

# df.plot(
#     ax=ax[0][0],
#     title='Line plot'
# )

# df.plot(
#     ax=ax[0][1],
#     style='o',
#     title='Scatter plot'
# )

# df.plot(
#     ax=ax[1][0],
#     kind='hist',
#     bins=50,
#     title='Histogram',
#     alpha=0.7
# )

# df.plot(
#     ax=ax[1][1],
#     kind='box',
#     title='Boxplot'
# )
# plt.tight_layout()

gt01 = df['data1']>0.1
piecounts = gt01.value_counts()
# print(piecounts)

# piecounts.plot(
#     kind='pie',
#     figsize=(5,5),
#     explode=[0,0.15],
#     labels=['<=0.1','>0.1'],
#     autopct='%1.1f%%',
#     shadow=True,
#     startangle=90,
#     fontsize=16
# )

# data=np.vstack(
#     [np.random.normal((0,0),2,size=[1000,2]),
#     np.random.normal((9,9),3,size=(2000,2))]
# )

# df = pd.DataFrame(data,columns=['x','y'])
# print(df.head())
# # df.plot()
# df.plot(
#     kind='hexbin',
#     x='x',
#     y='y',
#     bins=100,
#     cmap='rainbow'
# )

plt.show()