import cv2 as cv
import numpy as np
from os import system

def gamma(img, lamb):
     if lamb > 0:
          return np.power(img, 1/lamb)

imagem = cv.imread('imgs/macaco.png')

# Convertendo para o intervalo [0,1]
image01 = np.divide(imagem, 255)

# Aplicando correcao Gama
imgPower = gamma(image01, 2.5)

# Convertendo de volta para intervalo [0, 255]
resultado = np.multiply(imgPower, 255)


cv.imshow('image', imgPower)
cv.waitKey(0)
cv.destroyAllWindows()
system('clear') 



