import cv2 as cv
import numpy as np
import os

# imagem = input('Nome da imagem: ')
imagem = 'relogio.png'
# imagem = cv.imread('imagem.jpg')
imagemCinza= cv.imread(f'imgs/relogio.png', cv.IMREAD_GRAYSCALE)

# Suavizando imagem com gaussianBlue
imagemSuavizado = cv.GaussianBlur(imagemCinza, (21,21), 0)

# Constante para evitar divisão por 0 
constante = 1
imagemLapis = cv.divide(imagemCinza, imagemSuavizado + constante, scale=256)

# Imprimindo imagem na tela
cv.imshow('image', imagemLapis)
cv.waitKey(0)
cv.destroyAllWindows()
os.system('clear')