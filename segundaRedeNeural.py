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


#criando grafico para analise
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

cm = confusion_matrix(trainLabels,chuteAproximado)

def plot_confusion_matrix(
    cm,
    classes,
    normalize=False,
    title="Confusion_Matrix",
    cmap=plt.cm.Blues
    ):
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tickMarks= np.arange( len(classes)/2)
    plt.xticks(tickMarks,classes,rotation=45)
    plt.yticks(tickMarks,classes)
    

    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        print("Normalized Confusion Matrix")
    else:
        print("Confusion Matrix, without normalization")

    # print(cm)

    thresh = cm.max()/2
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,cm[i,j],
        horizontalalignment = 'center',
        color = 'white' if cm[i,j]>thresh else 'black'
        )
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

cmPlotLabels=['Sem efeitos','Com efeitos']
plot_confusion_matrix(cm,cmPlotLabels,title='Confusion Matrix')