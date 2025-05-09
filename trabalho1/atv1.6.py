import numpy as np
import cv2 as cv


imagem = cv.imread('imgs/macaco.png')

def planoBits(imagem, bit):
     mascara = np.power(2, bit)
     plano = np.where((imagem & mascara), 255, 0).astype(np.uint8)
     return plano

plano = planoBits(imagem, 6)

concatenado = cv.hconcat([imagem, plano])

cv.imwrite('imgs_geradas/atv1.6/source.png', concatenado)
