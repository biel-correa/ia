import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('/home/gabriel/Documentos/github/udemy/deveres/dever1/arquivos/weight-height.csv')

males = df.query('Gender == "Male"')
females = df.query('Gender == "Female"')

# data = [males,females]
img = plt.subplot()

img.set_title('Peso')
img.boxplot([males['Weight'],females['Weight']])


plt.legend(['Homens','Mulheres'])
plt.xticks('Homens','Mulheres')

plt.show()