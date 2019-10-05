import numpy as np
import keras
from keras import backend as k
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt


localTreinar = 'imgs/treinar'
localValidar = 'imgs/validar'
localTestar = 'imgs/testar'

batchesTreino = ImageDataGenerator().flow_from_directory(localTreinar,target_size=(224,224),classes=['cachorro','gato'],batch_size=10)
batchesTestar = ImageDataGenerator().flow_from_directory(localTestar,target_size=(224,224),classes=['cachorro','gato'],batch_size=10)
batchesValidar = ImageDataGenerator().flow_from_directory(localValidar,target_size=(224,224),classes=['cachorro','gato'],batch_size=6)

def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')
        
imgs,labels = next(batchesTreino)

plots(imgs,titles=labels)
teste sua mae e corna