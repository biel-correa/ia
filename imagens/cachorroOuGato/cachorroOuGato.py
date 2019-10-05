from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Activation,Dropout,Flatten,Dense
from keras import backend as K
import numpy as np 
from keras.preprocessing import image

imgLargura, imgAltura = 224,224

localTreinar = '/home/gabriel/Documentos/github/imagens/cachorroOuGato/imgs/treinar'
localValidar = '/home/gabriel/Documentos/github/imagens/cachorroOuGato/imgs/validar'

quantidadeTreino = 3997
quantidadeValidar = 6
epochs = 400
batch_size = 10

if K.image_data_format() == 'channels_first':
    input_shape = (3,imgLargura,imgAltura)
else:
    input_shape = (imgLargura,imgAltura,3)

treinarDatagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

testarDatagen = ImageDataGenerator(rescale=1./255)

treinarGerador = treinarDatagen.flow_from_directory(
    localTreinar,
    target_size=(imgLargura,imgAltura),
    batch_size=batch_size,
    class_mode='binary'
)

validarGerador = testarDatagen.flow_from_directory(
    localValidar,
    target_size=(imgLargura,imgAltura),
    batch_size=batch_size,
    class_mode='binary'
)

model = Sequential()
model.add(Conv2D(32,(3,3),input_shape=input_shape,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.summary()