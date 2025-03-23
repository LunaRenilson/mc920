import numpy as np
import cv2 as cv


imagem = cv.imread('imgs/macaco.png')

def negativa(imagem):
     negativa = np.subtract(255, imagem)
     return negativa

def restIntervalo(imagem, inicio, fim):
     intervalo = np.clip(imagem, inicio, fim).astype(np.uint8)
     return intervalo

def inverterPar(imagem):
     inversaPar = np.flip(imagem[::2, :], axis=1) # Invertendo as linhas pares da imagem
     imagem[::2, :] = inversaPar
     return imagem

def espelharMetade(imagem):
     altura = imagem.shape[0]
     metadeAltura = altura // 2
     imagem[metadeAltura:] = np.flip(imagem[:metadeAltura])
     return imagem

def espelhamentoVertical(imagem):
     imagem = np.flip(imagem)
     return imagem

resultado = espelhamentoVertical(imagem)

cv.imshow('image', resultado)
cv.waitKey(0)
cv.destroyAllWindows()