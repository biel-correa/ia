import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_circles

x, y = make_circles(
    n_samples=1000,
    noise=0.1,
    factor=0.2,
    random_state=0
)

#x.shape

# plt.figure(figsize=(5,5))
# #o = formato b = cor
# plt.plot(x[y==0,0],x[y==0,1],'ob',alpha=0.5)
# plt.plot(x[y==1,0],x[y==1,1],'xr',alpha=0.5)
# plt.xlim(-1.5,1.5)
# plt.ylim(-1.5,1.5)
# plt.legend(['0','1'])
# plt.title("Criar um grafico")
# #plt.show()

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

model = Sequential()
model.add(Dense(4,input_shape=(2,),activation='tanh'))
model.add(Dense(1,activation='sigmoid'))
model.compile(SGD(lr=0.5),'binary_crossentropy',metrics=['accuracy'])
model.fit(x,y,epochs=10,verbose=2)

hticks = np.linspace(-1.5,1.5,101)
vticks = np.linspace(-1.5,1.5,101)
aa, bb = np.meshgrid(hticks,vticks)
ab = np.c_[aa.ravel(),bb.ravel()]
c = model.predict(ab)
cc = c.reshape(aa.shape)

plt.figure(figsize=(5,5))
plt.contour(aa,bb,cc,cmap='bwr',alpha=0.2)
plt.plot(x[y==0,0],x[y==0,1],'ob',alpha=0.5)
plt.plot(x[y==1,0],x[y==1,1],'xr',alpha=0.5)
plt.xlim(-1.5,1.5)
plt.ylim(-1.5,1.5)
plt.legend(['0','1'])
plt.title('Criar grafico com informacoes dadas')
plt.show()