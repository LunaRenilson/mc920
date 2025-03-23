import numpy as np
import cv2 as cv


imagem = cv.imread('imgs/relogio.png')

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
     altura, largura = np.shape
     metadeAltura = altura // 2
     metadeImagem = imagem[:metadeAltura]


resultado = inverterPar(imagem)

cv.imshow('image', resultado)
cv.waitKey(0)
cv.destroyAllWindows()