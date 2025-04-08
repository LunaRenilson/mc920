import cv2 as cv
import numpy as np
import os

# imagem = cv.imread('imagem.jpg')
imagemCinza= cv.imread(f'imgs/relogio.png', cv.IMREAD_GRAYSCALE)

redimensionado = cv.resize(imagemCinza, (0, 0), fx = 0.5, fy = 0.5)

# Suavizando imagem com gaussianBlue
imagemSuavizado = cv.GaussianBlur(redimensionado, (31, 31), 0)
# Constante para evitar divisão por 0 
constante = 1
imagemLapis = cv.divide(redimensionado, imagemSuavizado + constante, scale=256)

concatenado = cv.hconcat([redimensionado, imagemLapis])

# Salvando imagem
cv.imwrite('./imgs_geradas/atv1.1/source.png', concatenado)