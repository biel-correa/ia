import numpy as np
from random import randint
from sklearn.preprocessing import MinMaxScaler

valores = []
menor500 = []

for i in range (1000):
    valor = randint(0,1000)
    valores.append(valor)
    if valores[i]<500:
        menor500.append(1)
    else:
        menor500.append(0)

valores = np.array(valores)

scaler = MinMaxScaler(feature_range = (0,1))
scalValores = scaler.fit_transform((valores).reshape(-1,1))

print (valores[0])
print (scalValores[0])
