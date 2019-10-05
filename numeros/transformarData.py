#https://www.youtube.com/watch?v=UkzhouEk6uY&list=PLZbbT5o_s2xrwRnXk_yCPtnqqo4_u2YGL&index=3
#sempre diminuir o maximo possivel do valor q sera usado, ajuda no processamento

import numpy as np 
from random import randint
from sklearn.preprocessing import MinMaxScaler

trainLabels = []
trainSamples = []

#gerar valores 
for i in range (1000):
    randomYounger = randint(13,64)
    trainSamples.append(randomYounger)
    trainLabels.append(0)

    randomOlder = randint(65,100)
    trainSamples.append(randomOlder)
    trainLabels.append(1)

for i in range (500):
    randomYounger = randint(13,64)
    trainSamples.append(randomYounger)
    trainLabels.append(1)

    randomOlder = randint(65,100)
    trainSamples.append(randomOlder)
    trainLabels.append(0)

#ver valores
# for i in trainSamples:
#     print(i)
#for i in trainLabels:
#    print (i)

trainLabels = np.array(trainLabels)
trainSamples = np.array(trainSamples)

scaler = MinMaxScaler(feature_range=(0,1))
scaledTrainSamples = scaler.fit_transform((trainSamples).reshape(-1,1))

print(trainSamples[0])
print(trainLabels[0])
print(scaledTrainSamples[0])