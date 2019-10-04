#https://www.youtube.com/watch?v=UkzhouEk6uY&list=PLZbbT5o_s2xrwRnXk_yCPtnqqo4_u2YGL&index=3
#sempre diminuir o maximo possivel do valor q sera usado, ajuda no processamento

import numpy as np 
from random import randint
from sklearn.preprocessing import MinMaxScaler

#gerar novos valores
trainLabels = []
trainSamples = [] 
for i in range (10):
    randomYounger = randint(13,64)
    trainSamples.append(randomYounger)
    trainLabels.append(1)

    randomOlder = randint(65,100)
    trainSamples.append(randomOlder)
    trainLabels.append(0)

for i in range (200):
    randomYounger = randint(13,64)
    trainSamples.append(randomYounger)
    trainLabels.append(0)

    randomOlder = randint(65,100)
    trainSamples.append(randomOlder)
    trainLabels.append(1)


trainLabels = np.array(trainLabels)
trainSamples = np.array(trainSamples)

scaler = MinMaxScaler(feature_range=(0,1))
scaledTrainSamples = scaler.fit_transform((trainSamples).reshape(-1,1))

import keras
from keras import backend as k 
from keras.models import Sequential
from keras.layers import activations
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.models import model_from_json

jsonFile = open('redesTreinadas/remedio.json','r')
modeloCarregadoJson = jsonFile.read()
jsonFile.close()
model = model_from_json(modeloCarregadoJson)
model.load_weights('redesTreinadas/remedio.h5')
print('\n carregado \n')

chute = model.predict(
    scaledTrainSamples,
    batch_size=10,
    verbose=1
)

chuteAproximado = model.predict_classes(
    scaledTrainSamples,
    batch_size=10,
    verbose=1
)
print(chute[0])
print(chuteAproximado[0])

#model.summary()
