import numpy as np
import cv2 as cv


imagem = cv.imread('imgs/macaco.png')

def negativa(imagem):
     neg = np.subtract(255, imagem)
     concatenado = cv.hconcat([imagem, neg])
     cv.imwrite('imgs_geradas/atv1.8/negativa.png', concatenado)
     return negativa

def restIntervalo(imagem, inicio, fim):
     intervalo = np.clip(imagem, inicio, fim).astype(np.uint8)
     concatenado = cv.hconcat([imagem, intervalo])
     cv.imwrite('imgs_geradas/atv1.8/intervaloRestrito.png', concatenado)
     return intervalo

def inverterPar(imagem):
     invertido = imagem.copy()
     inversaPar = np.flip(invertido[::2, :], axis=1) # Invertendo as linhas pares da imagem
     invertido[::2, :] = inversaPar
     concatenado = cv.hconcat([imagem, invertido])
     cv.imwrite('imgs_geradas/atv1.8/invertido.png', concatenado)
     return imagem

def espelharMetade(imagem):
     espelhado = imagem.copy()
     altura = espelhado.shape[0]
     metadeAltura = altura // 2
     espelhado[metadeAltura:] = np.flip(espelhado[:metadeAltura])
     concatenado = cv.hconcat([imagem, espelhado])
     cv.imwrite('imgs_geradas/atv1.8/espelhamentoMetade.png', concatenado)

def espelhamentoVertical(imagem):
     espelhada = np.flip(imagem)
     concatenado = cv.hconcat([imagem, espelhada])
     cv.imwrite('imgs_geradas/atv1.8/espelhamentoVertical.png', concatenado)

negativa(imagem)
restIntervalo(imagem, 100, 200)
inverterPar(imagem)
espelharMetade(imagem)
espelhamentoVertical(imagem)