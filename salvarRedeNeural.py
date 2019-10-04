modelJson = model.to_json()
with open("nome.json","w") as json_file:
    json_file.write(modelJson)
model.save_weights("nome.h5")
print("salvo")