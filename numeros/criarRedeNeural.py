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

model.summary()

model.compile(Adam(lr=.0001),loss="sparse_categorical_crossentropy",metrics=['accuracy'])