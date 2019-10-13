import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('/home/gabriel/Documentos/github/udemy/arquivos/weight-height.csv')
df.plot(
    kind='scatter',
    x='Height',
    y='Weight',
    title='Peso X Altura'
)
# linha desenhada a mao
plt.plot([55,78],[75,250],color='purple',linewidth=3)

def line(x,w=0,b=0):
    return x*w+b
x = np.linspace(55,80,100)
yhat = line(x,w=0,b=0)

df.plot(
    kind='scatter',
    x='Height',
    y='Weight',
    title='Peso X Altura'
)

plt.plot(x,yhat,color='red',linewidth=3)

def meanSquaredError(yTrue,yPred):
    s = (yTrue - yPred)**2
    return s.mean()
x=df[['Height']].values
yTrue = df['Weight'].values
yPred = line(x)
meanSquaredError(yTrue,yPred)

plt.figure(figsize=(9,4))

ax1 = plt.subplot(121)
df.plot(
    kind='scatter',
    x='Height',
    y='Weight',
    title='Peso X Altura',
    ax=ax1
)

bbs = np.array([-100,-50,0,50,100,150])
mses = []
for b in bbs:
    yPred = line(x,w=2,b=b)
    mse = meanSquaredError(yTrue,yPred)
    mses.append(mse)
    plt.plot(x,yPred)

ax2 = plt.subplot(122)
plt.plot(bbs,mses,'o-')
plt.title('Cost function of b')
plt.xlabel('b')

plt.show()