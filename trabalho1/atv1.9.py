import numpy as np
import cv2 as cv


imagem = cv.imread('imgs/macaco.png')

def quantizacao(imagem, niveis):
     if niveis >= 2 and niveis < 256: 
          intervalo = np.divide(255, (niveis - 1)) # Obtendo intervalo reduzido
          quantizada = np.round(imagem / intervalo) * intervalo  # Quantização
          resultado = quantizada.astype(np.uint8)
          return resultado

resultado = quantizacao(imagem, 2)

concatenado = cv.hconcat([imagem, resultado])
cv.imwrite('imgs_geradas/atv1.9/source.png', concatenado)