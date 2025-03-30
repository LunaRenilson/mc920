import numpy as np
import cv2 as cv


imagem = cv.imread('imgs/macaco.png')

resultado = np.histogram(imagem, bins=256)

print(resultado)

# cv.imshow('image', resultado)
# cv.waitKey(0)
# cv.destroyAllWindows()