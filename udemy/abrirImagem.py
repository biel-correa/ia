from PIL import Image
import numpy as np

img = Image.open('/home/gabriel/Documentos/github/udemy/arquivos/iss.jpg')

imgArray = np.asarray(img)
print(imgArray.shape)
print(imgArray.ravel().shape)