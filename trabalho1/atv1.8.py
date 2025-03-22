import numpy as np
import cv2 as cv


imagem = cv.imread('imgs/macaco.png')

negativa = np.subtract(255, imagem)
intervalo = np.clip(imagem, 100, 200).astype(np.uint8)

cv.imshow('image', intervalo)
cv.waitKey(0)
cv.destroyAllWindows()