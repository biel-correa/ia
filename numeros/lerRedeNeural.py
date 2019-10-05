from keras.models import model_from_json
jsonFile = open('nome.json','r')
modeloCarregadoJson = jsonFile.read()
jsonFile.close()
model = model_from_json(modeloCarregadoJson)
model.load_weights('nome.h5')
print('\n carregado \n')

# ou 

from keras.models import load_model

new_model = load_model('redesTreinadas/teste.h5')