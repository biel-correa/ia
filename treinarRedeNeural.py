#https://www.youtube.com/watch?v=UkzhouEk6uY&list=PLZbbT5o_s2xrwRnXk_yCPtnqqo4_u2YGL&index=3
#sempre diminuir o maximo possivel do valor q sera usado, ajuda no processamento

import numpy as np 
from random import randint
from sklearn.preprocessing import MinMaxScaler

trainLabels = []
trainSamples = []

#gerar valores 
for i in range (50):
    randomYounger = randint(13,64)
    trainSamples.append(randomYounger)
    trainLabels.append(1)

    randomOlder = randint(65,100)
    trainSamples.append(randomOlder)
    trainLabels.append(0)

for i in range (1000):
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

# print(trainSamples[0])
# print(trainLabels[0])
# print(scaledTrainSamples[0])

import keras
from keras import backend as k 
from keras.models import Sequential
from keras.layers import activations
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

model = Sequential()

model.add(Dense(16,input_shape=(1,),activation="relu"))
model.add(Dense(32,activation="relu"))
model.add(Dense(2,activation="softmax"))

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=Adam(lr=.0001),
    metrics=['accuracy']
)

model.fit(
    scaledTrainSamples,
    trainLabels,
    validation_split=0.1,
    batch_size=10,
    epochs=1000,
    shuffle= True,
    verbose=2
)

modelJson = model.to_json()
with open("remedio.json","w") as json_file:
    json_file.write(modelJson)
model.save_weights("remedio.h5")
print("salvo")


#model.summary()