from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('/home/gabriel/Documentos/github/udemy/arquivos/weight-height.csv')
x=df[['Height']].values
yTrue = df['Weight'].values


model = Sequential()
model.add(Dense(1,input_shape=(1,)))
model.summary()
model.compile(Adam(lr=0.8),'mean_squared_error')
model.fit(x,yTrue,epochs=40)

yPred = model.predict(x)

df.plot(
    kind='scatter',
    x='Height',
    y='Weight',
    title='Peso e altura'
)

plt.plot(x,yPred,color='red')

plt.show()

w,b = model.get_weights()